# Loss Analysis and Debugging Summary

## Problem
During prediction, the logits are the same, whatever the images.  
**Loc Loss**: decreases  
**Conf Loss**: decreases, but quickly converge  
<img src="_05_inference/exp0.png" width="400" alt="Inference Result">


## Prediction Confidence Statistics Example

| Sample       | Highest Foreground Score | Mean Foreground Score | Foreground > 0.5 Count |
|--------------|---------------------------|------------------------|------------------------|
| First Sample | 0.2757                   | 0.2044                | 0                      |
| Second Sample| 0.2159                   | 0.1428                | 0                      |

## Trials

- **Identicalize the pre-processing of images while prediction**  
  Addressed potential image differences and model misperformance caused by inconsistent normalization/scaling between training and inference.

- **Double check the shape**  
  Verified tensor dimensions; confirmed there was no logical error in how data was being passed or reshaped.

- **Matcher threshold (IOU)**  
  Lowered to increase positive examples, giving the model more "target" anchors to learn from during each iteration.

- **Image augmentation**  
  Evaluated if current methods were clearing the boxes in the image or resulting in too little positive examples for the model to digest.

- **Prior Probability Initialization**  
  Found to have no major impact on final accuracy; while the loss truly decreases faster, it still resulted in false predictions, suggesting the model might be stuck in a regional optimal.

- **Hard Negative Mining (HNM), change ratio**  
  Attempted to adjust the balance, though the score fluctuated and remained low. Investigated if the negative examples were so numerous that they caused false classification.

- **Alpha (weight of loc to conf loss)**  
  Changed the ratio but saw no improvement. Incorrectly assumed the classification (conf) worked because it converged, while the localization (loc) did not.

- **Lower confidence while prediction**  
  Tested if the model simply wasn't trained enough, resulting in valid detections that were being filtered out by a high threshold.

- **Add batch normalization**  
  Check if its caused by the dead model. (The most possible reason)  
  Relu → negative neurons, zero gradient, the weight is dead, only bias functions.  
  The input itself loses the features through the network. It only learns to output the ratio of the median of positive and negative examples based on HNM.  
  Outputs the same logits at all time.  
  **BUT!!! It fails again!** Though the loc loss converges faster and better. This is on the backbone model itself.  
  ![alt text](_04_training/loss/exp/exp9.png)

- **Add prior probability bias and Xavier**  
  Double checked the json, that the categories only contain the labels that are trained.  
  ![alt text](_04_training/loss/exp/exp10.png)


## Final Solution
- **Add one more layer on the feature map of each stage.**
- **Guarantees processing feature map data thoroughly. The backbone structure is only for producing feature map for larger receptive field**  
![alt text](_04_training/loss/exp/exp11.png)

# Later Experiments and Adjustments

**12.** Add more layers on the feature map, create config file.  
![alt text](_04_training/loss/exp/exp12.png)
###

**13.** 
Increases epochs, NEG_RATIO, OVERLAP_THRESHOLD, add milestones in scheduler.  
→ **Overfitting!**  
![alt text](_04_training/loss/exp/exp13.png)
###

**14.**  
Shrink model capacity: halving the stage channels, removing one convolutional layer from the detection heads.  
Inject noise and regularization: adding Dropout2d to the heads, implementing gradient clip, applying weight decay.  
Expand the positive dataset: increasing AUG_FACTOR, lowering the OVERLAP_THRESHOLD.  
Stabilize the classification head: reducing the NEG_RATIO.  
→ **Underfitting!**  
The training process is too time-consuming, validation loss exceeds the training loss.  
![alt text](_04_training/loss/exp/exp14.png)
<img src="_05_inference/exp14.png" width="400" alt="Inference Result">
###

**15.** Finalize the file structure, lower dropout and weight decay, add early stop logic and workflow diagram.  
→ **Underfitting!**  
The model has already learned all the things it can.  
<img src="_04_training/loss/exp/exp15.png" width="600" alt="Training Loss">
<img src="_05_inference/exp15.png" width="400" alt="Inference Result">
###

**16.**  
Increase model performance: incrase STAGE_CHANNELS and EARLY_STOP_PATIENCE, lower weight decay and dropout.  
Increase data diversity: add more complex images transform.  
Stop meaningless training: lower epochs, SCHEDULER_MILESTONES.
→ **Underfitting!**  even worse, too heavy for a small model
![alt text](_04_training/loss/exp/exp16.png)

**17.**  
Comment out the extra augmentation actions (add GaussianBlur), lower NMS_OVERLAP_THRESHOLD.  
→ **Almost Perfect in Inference** for single model despite the loss curve.  
The model has already tried its full performance and make full use of the limited data. Train all folds.
![alt text](_04_training/loss/loss_1.png)
![alt text](_04_training/loss/loss_2.png)
![alt text](_04_training/loss/loss_3.png)
![alt text](_04_training/loss/loss_4.png)
![alt text](_04_training/loss/loss_5.png)
Change inference logic to multi-model. Experiment with different model and parameter combination. Optimize prediction. Finalize all files.  
- **Most predicted boxes are at same scale -> can improve in loc prediction in w and h.**  
- **Sensitive to light changes -> need higher probability of image augmentation, doesn't generalize well.**  
- **Likely to recognize elbow and face as fist -> false prediction, eed more negative examples and image augmentation.**  
<img src="_05_inference/exp17_1.png" width="250" alt="Inference Result">
<img src="_05_inference/exp17_2.png" width="250" alt="Inference Result">
<img src="_05_inference/exp17_3.png" width="250" alt="Inference Result">
<img src="_05_inference/exp17_4.png" width="250" alt="Inference Result">
<img src="_05_inference/exp17_5.png" width="250" alt="Inference Result">
<img src="_05_inference/exp17_6.png" width="250" alt="Inference Result">

# Next Step
## For this project
- **Add more data!!!**
- **Try better parameter and model combination**
- **This lightweight model needs more complex structure**

## For other projects (reflections)
## Reflections
### Hard Negative Mining
Among the 105,000 boxes, the vast majority are background.  
The positive-to-negative ratio can reach 1:1000. Before calculating loss, you need to sort background samples by loss in descending order and select only 3 times the number of positive samples as negatives (ensuring a 1:3 ratio). Otherwise the model will learn to output only background predictions.

### Industrial Project Mindsets

#### 1. Deconstructive Thinking: Everything is "Configuration"
In industrial projects, `main.py` rarely contains hardcoded numbers (like learning rate 0.001 or input size 300x300).  

**Mindset shift**: Don't mix logic and parameters together.  
**Approach**: Create a `config.py` or use a YAML file.  
**Benefit**: When results are not good during training, you can simply change Anchor sizes, learning rate, or other parameters in the config file without touching the core logic code. This keeps the main code clean.

#### 2. Error Prevention Thinking: Checkpoint Resumption
Industrial training often runs for days or weeks. Power cuts or interruptions can happen.  

**Mindset shift**: Never assume the program will finish without interruption.  
**Approach**: At the end of each epoch, save both model weights (`model.state_dict()`) and optimizer state (`optimizer.state_dict()`).  
**Benefit**: You can resume training precisely from any epoch (e.g. epoch 50) instead of starting over.

#### 3. Monitoring Thinking: Don't Just Look at Loss Numbers
Scrolling numbers in the terminal are hard to read and understand.  

**Mindset shift**: Humans understand curves and visuals much better than raw numbers.  
**Approach**: Use TensorBoard or WandB.  
**Benefit**: You can monitor loss curves in real-time. A flat curve may indicate gradient vanishing; a sudden jump to NaN usually means the learning rate is too high. You can even visualize model predictions on sample images during training.

#### 4. Data Poisoning Prevention: Data Validation (Sanity Check)
"Garbage in, garbage out."  

**Mindset shift**: Never fully trust your dataset.  
**Approach**: Before starting the formal training loop, write a small script to load 10 images from the DataLoader and draw the ground truth boxes on them for visual inspection.  
**Benefit**: You will often discover issues like reversed or wrong normalized coordinates that make the model waste time learning nothing.

#### 5. Inference Thinking: Not Just Training
The real goal of industrial projects is deployment, not just minimizing loss.  

**Mindset shift**: Training code and inference code should be clearly separated.  
**Approach**:  
- Training mode: includes matching, Hard Negative Mining, and `Loss.backward()`  
- Eval/Inference mode: remove all loss-related logic, add decode + NMS  
**Benefit**: Inference becomes much faster (higher FPS) because all training-only computations are removed.

### Structured Project Naming Convention
Use meaningful folder names that follow logical order instead of numbers like `01`, `02`. VS Code sorts files alphabetically, so arrange them smartly:

- `assets/` — pictures and data (always on top)  
- `configs/` — configuration files  
- `core/` — core model and training code  
- `scripts/` — training and running scripts  
- `utils/` — helper functions and classes  

This convention is both clear and practical.

### Common Debugging Issues & Solutions

#### 1. Runtime Crashes on Windows ("Process Escape")
**Problem**: Program crashes immediately or after a few epochs with `PicklingError` or `DataLoader worker exited unexpectedly`.  

**Root cause**: Windows uses `spawn` for multi-processing, which re-imports the whole script and can cause infinite loops.  

**Solution**:  
- Add `if __name__ == '__main__':` protection.  
- Call `gc.collect()` and `torch.cuda.empty_cache()` after each fold to free memory.

#### 2. Background Class Identity Crisis
**Problem**: `Index out of range` error or extremely high Conf Loss. Model treats everything as background.  

**Root cause**: In SSD-style detectors, class 0 is reserved for background. If your target class starts from 0 in the dataset, it conflicts.  

**Solution**:  
- Add +1 to all labels when loading the dataset.  
- Make sure `num_classes` in loss includes background (target classes + 1).

#### 3. Low GPU Utilization (GPU Waiting for Data)
**Problem**: GPU usage fluctuates wildly and `it/s` is very low.  

**Root cause**: Data loading and augmentation on CPU is too slow (IO bottleneck).  

**Solution**:  
- Use larger `aug_factor` to pre-augment and cache more data in memory (trade memory for speed).  
- Add cache refresh logic every few epochs to keep data fresh while maintaining high GPU utilization.

#### 4. Task Imbalance (Good Classification, Poor Localization)
**Problem**: Conf Loss converges quickly, but Loc Loss stays high. Predicted boxes are unstable in size.  

**Root cause**: Classification is too easy compared to regression, so gradients are dominated by classification.  

**Solution**:  
- Dynamically increase `alpha` (weight of localization loss) in later epochs.  
- Raise the regression loss weight (e.g. to 2.0) to force the model to focus more on accurate box positioning.

#### 5. Monitoring Visualization: Manual vs Digital
**Problem**: Manually opening many loss images and comparing experiments is inefficient.  

**Solution**: Use TensorBoard or WandB for live monitoring, easy multi-experiment comparison, and visual debugging.

**Advanced tip**: Consider using pretrained backbone models with fine-tuning to speed up convergence.