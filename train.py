import torch
import torch.optim as optim
from _01_data.dataloader import get_k_fold_data
from _03_model import LargeScaleSSD, MultiBoxLoss, create_prior_boxes
from _00_config.config import Config
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

# --- Core Protection: Windows multi-threading must be placed in this check ---
if __name__ == '__main__':
    # 1. Initialize model and loss
    device = Config.DEVICE
    priors = create_prior_boxes().to(device) # Create priors and place on device
    criterion = MultiBoxLoss(num_classes=Config.NUM_CLASSES).to(device)

    # 2. Load data
    K = Config.K_FOLD
    for fold in range(K):
        print(f"\n" + "="*20 + f" Starting Fold {fold+1}/{K} Training " + "="*20)
        
        # 0. Clear GPU memory from previous fold
        if fold > 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 1. Get new data for each fold
        train_loader, val_loader = get_k_fold_data(
            ann_file=Config.ANN_FILE,
            image_root=Config.IMAGE_ROOT,
            batch_size=Config.BATCH_SIZE,
            k=K,
            i=fold 
        )

        # 2. Reinitialize model and optimizer for each fold
        net = LargeScaleSSD(num_classes=Config.NUM_CLASSES).to(device)
        optimizer = optim.Adam(net.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        scheduler = MultiStepLR(optimizer, milestones=Config.SCHEDULER_MILESTONES, gamma=Config.SCHEDULER_GAMMA)
        criterion.alpha = Config.ALPHA

        # Record for plotting
        history = {
            'train_loss': [], 'train_loss_l': [], 'train_loss_c': [],
            'val_loss': [], 'val_loss_l': [], 'val_loss_c': [], 'actual_loss': []
        }
        best_val_loss = float('inf')

        total_steps = Config.EPOCHS * len(train_loader)
        pbar = tqdm(total=total_steps, desc=f"Fold {fold+1}", unit="batch", leave=True)
        no_improve_count = 0

        for epoch in range(Config.EPOCHS):
            # --- Training Phase ---
            net.train()
            train_running_loss = 0
            train_running_loss_l = 0.0
            train_running_loss_c = 0.0
            for images, targets in train_loader:
                images = torch.stack(images).to(device)
                targets = [t.to(device) for t in targets]
                
                optimizer.zero_grad()
                out = net(images)
                loss_l, loss_c, total_loss = criterion(out, targets, priors)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), Config.GRAD_CLIP)
                optimizer.step()
                train_running_loss += total_loss.item()
                train_running_loss_l += loss_l.item()
                train_running_loss_c += loss_c.item()
                pbar.update(1)

            # --- Validation Phase ---
            net.eval() 
            val_running_loss = 0
            val_running_loss_l = 0.0
            val_running_loss_c = 0.0
            val_actual_loss = 0.0
            with torch.no_grad(): 
                for images, targets in val_loader:
                    images = torch.stack(images).to(device)
                    targets = [t.to(device) for t in targets]
                    out = net(images)
                    v_loss_l, v_loss_c, v_total_loss = criterion(out, targets, priors)
                    val_running_loss_l += v_loss_l.item()
                    val_running_loss_c += v_loss_c.item()
                    val_running_loss += v_total_loss.item()
                    val_actual_loss += (v_loss_l.item() + v_loss_c.item())

            # Compute average loss
            avg_train_loss = train_running_loss / len(train_loader)
            avg_val_loss = val_running_loss / len(val_loader)
            avg_actual_loss = val_actual_loss / len(val_loader)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_loss_l'].append(train_running_loss_l / len(train_loader))
            history['train_loss_c'].append(train_running_loss_c / len(train_loader))
            history['val_loss_l'].append(val_running_loss_l / len(val_loader))
            history['val_loss_c'].append(val_running_loss_c / len(val_loader))
            history['actual_loss'].append(avg_actual_loss)

            if epoch > 0 and epoch % Config.CACHE_REFRESH_FREQUENCY == 0:
                train_loader.dataset.refresh_cache(pbar=pbar)

            if epoch % Config.PRINT_FREQUENCY == 0 or epoch == Config.EPOCHS - 1:
                pbar.write(f"Fold {fold+1} | Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if epoch > Config.EPOCHS - Config.BEST_MODEL_SAVE_EPOCHS:
                if avg_actual_loss < (best_val_loss - 1e-4):
                    best_val_loss = avg_actual_loss
                    no_improve_count = 0
                    best_path = os.path.join('_04_training/models', f"best_{fold+1}.pth")
                    torch.save(net.state_dict(), best_path)
                    pbar.write(f"Fold {fold+1} | Epoch {epoch+1 } | Best Val Loss: {avg_val_loss:.4f} | Best Model Saved")
                else:
                    no_improve_count += 1
            
            if no_improve_count >= Config.EARLY_STOP_PATIENCE:
                pbar.write(f"Early stop triggered at epoch {epoch+1}")
                break
            
            scheduler.step()

        latest_path = os.path.join('_04_training/models', f"latest_{fold+1}.pth")
        torch.save(net.state_dict(), latest_path)
        pbar.write(f"Fold {fold+1} | Epoch {epoch+1 } | Latest Val Loss: {avg_val_loss:.4f} | Latest Model Saved")
        pbar.close()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        # Left plot: Train vs Val (Total Loss)
        ax1.plot(history['train_loss'], color='blue', label='Train Total')
        ax1.plot(history['val_loss'], color='red', linestyle='--', label='Val Total')
        ax1.plot(history['actual_loss'], color='purple', linestyle='-.', label='Val Actual')
        ax1.set_title(f'Fold {fold+1} Total Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Right plot: Classification vs Localization (Key metrics)
        ax2.plot(history['train_loss_l'], color='green', label='Train Loc')
        ax2.plot(history['train_loss_c'], color='orange', label='Train Conf')
        # Validation details can also be added, but using dashed lines
        ax2.plot(history['val_loss_l'], color='green', linestyle='--', alpha=0.8, label='Val Loc')
        ax2.plot(history['val_loss_c'], color='orange', linestyle='--', alpha=0.8, label='Val Conf')
        ax2.set_title(f'Fold {fold+1} Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout() # Prevent title overlap
        plt.savefig(os.path.join('_04_training/loss', f'loss_{fold+1}.png'))
        plt.close(fig) 