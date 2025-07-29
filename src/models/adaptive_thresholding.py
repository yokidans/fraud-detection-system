import pickle
from collections import deque
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression

class AdaptiveThreshold:
    def __init__(self, initial_threshold: float = 0.5, 
                 memory_size: int = 1000,
                 min_threshold: float = 0.1,
                 max_threshold: float = 0.9):
        self.initial_threshold = initial_threshold
        self.memory_size = memory_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.feedback_memory = deque(maxlen=memory_size)
        self.cost_matrix = {
            'false_positive': 1.0,  # Cost of rejecting a legitimate transaction
            'false_negative': 5.0   # Cost of accepting a fraudulent transaction
        }
    
    def get_adjusted_threshold(self, probability: float) -> float:
        """Get the current optimal threshold"""
        if not self.feedback_memory:
            return self.initial_threshold
        
        # Calculate empirical costs at different thresholds
        thresholds = np.linspace(0, 1, 101)
        costs = []
        
        for threshold in thresholds:
            fp, fn = 0, 0
            for feedback in self.feedback_memory:
                pred = feedback['probability'] >= threshold
                if pred and not feedback['actual']:
                    fp += 1
                elif not pred and feedback['actual']:
                    fn += 1
            
            cost = (fp * self.cost_matrix['false_positive'] + 
                    fn * self.cost_matrix['false_negative'])
            costs.append(cost)
        
        optimal_idx = np.argmin(costs)
        return np.clip(thresholds[optimal_idx], self.min_threshold, self.max_threshold)
    
    def update(self, actual: int, predicted: int, probability: float):
        """Update the threshold based on new feedback"""
        self.feedback_memory.append({
            'actual': actual,
            'predicted': predicted,
            'probability': probability
        })
    
    def save(self, filepath: str):
        """Save the thresholder state to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'initial_threshold': self.initial_threshold,
                'memory': list(self.feedback_memory),
                'cost_matrix': self.cost_matrix
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load a thresholder from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        thresholder = cls(
            initial_threshold=data['initial_threshold'],
            memory_size=len(data['memory'])
        )
        thresholder.feedback_memory = deque(data['memory'], maxlen=thresholder.memory_size)
        thresholder.cost_matrix = data.get('cost_matrix', thresholder.cost_matrix)
        return thresholder