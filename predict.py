import torch
import cv2
from torchvision.transforms import v2
from torchvision import tv_tensors
from _03_model import LargeScaleSSD, create_prior_boxes
from _02_utils import nms, decode
from _00_config.config import Config
import json
import PIL.Image as Image


def load_ensemble_models(weight_paths, num_classes, device):
    models = []
    for weight_path in weight_paths:
        net = LargeScaleSSD(num_classes=num_classes).to(device)
        net.load_state_dict(torch.load(weight_path, map_location=device))
        net.eval()
        models.append(net)
        print(f"Loaded model from {weight_path}")
    return models


def ensemble_forward(models, img, device):
    all_loc_data = []
    all_conf_data = []
    
    with torch.no_grad():
        for model in models:
            loc_data, conf_data = model(img)
            all_loc_data.append(loc_data)
            all_conf_data.append(conf_data)
    
    loc_data_concat = torch.cat(all_loc_data, dim=1)   # [1, num_models * num_anchors, 4]
    conf_data_concat = torch.cat(all_conf_data, dim=1) # [1, num_models * num_anchors, num_classes]
    
    return loc_data_concat, conf_data_concat


def camera_detect():
    # 1. Environment and path configuration
    device = Config.DEVICE
    weight_paths = Config.WEIGHT_PATHS
    label_path = Config.LABEL_MAP_FILE

    # 2. Load label mapping and add background class
    with open(label_path, "r") as f:
        labels_data = json.load(f)

    label_map = {0: "background"}

    if isinstance(labels_data, dict):
        for k, v in labels_data.items():
            label_map[int(k) + 1] = v
    elif isinstance(labels_data, list):
        for item in labels_data:
            label_map[item["id"] + 1] = item["name"]

    num_classes = len(label_map)
    num_models = len(weight_paths)

    # 3. Initialize ensemble of models and priors
    models = load_ensemble_models(weight_paths, num_classes, device)
    priors = create_prior_boxes().to(device)

    # Expand priors to match ensemble concatenated anchors
    # [num_models * num_anchors, 4]
    priors_expanded = priors.repeat(num_models, 1)

    # 4. Start camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)

    print(f"Ensemble Detection started with {len(models)} models! Current classes: {label_map}")

    eval_transform = v2.Compose([
        v2.Resize(size=Config.INPUT_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=Config.NORMALIZATION_MEAN, std=Config.NORMALIZATION_STD)
    ])

    def color_for(cid):
        palette = Config.DETECTION_COLORS
        return palette[(cid - 1) % len(palette)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # --- Preprocessing ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        t_img = tv_tensors.Image(img_pil)
        img = eval_transform(t_img).unsqueeze(0).to(device)

        # --- Ensemble Inference ---
        loc_data, conf_data = ensemble_forward(models, img, device)

        # --- Post-processing (fully vectorized) ---
        conf_preds = torch.softmax(conf_data, dim=-1)[0]  # [total_anchors, num_classes]
        loc_preds = loc_data[0]                            # [total_anchors, 4]

        # Get best non-background score and class per anchor
        scores_all, cls_ids_all = conf_preds[:, 1:].max(dim=1)  # [total_anchors]

        # Threshold filter via boolean mask
        mask = scores_all > Config.CONF_THRESHOLD

        if mask.sum() == 0:
            final_detections = []
        else:
            filtered_locs   = loc_preds[mask]           # [K, 4]
            filtered_priors = priors_expanded[mask]     # [K, 4]
            filtered_scores = scores_all[mask]          # [K]
            filtered_cls    = cls_ids_all[mask] + 1     # [K], +1 to restore background offset

            # Decode boxes
            filtered_boxes = decode(filtered_locs, filtered_priors, Config.VARIANCES)

            # NMS
            keep_indices = nms(
                filtered_boxes,
                filtered_scores,
                overlap=Config.NMS_OVERLAP_THRESHOLD,
                top_k=Config.NMS_TOP_K
            )

            # Assemble results
            final_detections = []
            for i in keep_indices:
                idx = i.item() if torch.is_tensor(i) else i
                final_detections.append({
                    'box':    filtered_boxes[idx].cpu().numpy(),
                    'score':  filtered_scores[idx].item(),
                    'cls_id': filtered_cls[idx].item()
                })

        if len(final_detections) > 0:
            print(f"Detections: {len(final_detections)}")

        # --- Draw boxes ---
        for det in final_detections:
            box_n  = det['box']
            score  = det['score']
            cls_id = det['cls_id']

            x1, y1, x2, y2 = (box_n * [w, h, w, h]).astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            color = color_for(cls_id)
            label = label_map.get(cls_id, f"cls{cls_id}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("SSD Ensemble Detect", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_detect()