import pytest
import torch
import torch.nn as nn
from traffic_scene_graph.models.relation_classifier import RelationClassifier
from traffic_scene_graph.training.classification_loss import MultiTaskLoss
from traffic_scene_graph.training.trainer import SelfTrainingScheduler

def test_relation_classifier():
    classifier = RelationClassifier(input_dim=32, hidden_dim=64, num_classes=8)
    
    x = torch.randn(10, 32)
    logits = classifier(x)
    
    assert logits.shape == (10, 8)

def test_multi_task_loss():
    loss_fn = MultiTaskLoss(num_classes=8, label_smoothing=0.0)
    
    contrastive_loss = torch.tensor(1.0)
    relation_logits = torch.randn(5, 8)
    # pseudo labels: [0, 1, 7, 2, 4] -> class 7 is UNKNOWN
    pseudo_labels = torch.tensor([0, 1, 7, 2, 4])
    
    loss_dict = loss_fn(contrastive_loss, relation_logits, pseudo_labels, alpha=0.5)
    
    assert "total" in loss_dict
    assert "contrastive" in loss_dict
    assert "classification" in loss_dict
    
    # We should have loss from classification for non-7 labels (4 labels valid)
    assert loss_dict["classification"] > 0

def test_self_training_scheduler():
    scheduler = SelfTrainingScheduler(total_epochs=100, stage1_ratio=0.2, stage2_ratio=0.5)
    
    # Epoch 10: Rule based
    assert scheduler.get_training_mode(10) == "rule_based"
    assert scheduler.get_classification_weight(10) == 0.1
    
    # Epoch 30: Confident prediction
    assert scheduler.get_training_mode(30) == "confident_prediction"
    
    # Epoch 80: Refinement
    assert scheduler.get_training_mode(80) == "refinement"
    assert scheduler.get_classification_weight(80) == 1.5
    
    # Test refinement logic
    rule_labels = torch.tensor([0, 7, 1])
    class_probs = torch.tensor([
        [0.1, 0.9, 0.0], # Class 1 confident
        [0.0, 0.0, 0.9], # Class 2 confident
        [0.4, 0.4, 0.2]  # Unconfident
    ])
    # For categorical classification we provide logits. So:
    model_logits = torch.log(class_probs + 1e-6)
    
    # stage 1
    labels_s1 = scheduler.refine_labels(rule_labels, model_logits, epoch=10)
    assert torch.equal(labels_s1, torch.tensor([0, 7, 1]))
    
    # stage 2 (overwrites UNKNOWN)
    labels_s2 = scheduler.refine_labels(rule_labels, model_logits, epoch=30)
    assert torch.equal(labels_s2, torch.tensor([0, 2, 1]))
    
    # stage 3 (overwrites all if confident)
    labels_s3 = scheduler.refine_labels(rule_labels, model_logits, epoch=80)
    # The first one was 0, but prob is 0.9 for class 1 -> 1
    # The second was 7, prob 0.9 for class 2 -> 2
    # The third was 1, unconfident -> 1
    assert torch.equal(labels_s3, torch.tensor([1, 2, 1]))
