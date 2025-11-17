# Online Learning: Adapt from Production Runs

**Enhancement**: #11 - Continual Learning from Deployment  
**Difficulty**: High  
**Impact**: Medium  
**Priority**: 10

---

## Problem Statement

**Current**: RL agent frozen after training.

**Problem**:
- Real-world problems may differ from training distribution
- User preferences evolve over time
- New constraints added in production

**Opportunity**: Learn from production scheduling runs to improve policy continuously.

---

## Solution: Online Learning

### Architecture

```python
class OnlineLearningSystem:
    def __init__(self, base_policy):
        self.policy = base_policy
        self.experience_buffer = []
        self.update_frequency = 100  # Update every 100 production runs
    
    def schedule_problem(self, problem):
        """
        Generate schedule AND collect experience.
        """
        # 1. Run GA with current policy
        schedule, experience = run_ga_with_experience_collection(
            problem, self.policy
        )
        
        # 2. Store experience
        self.experience_buffer.extend(experience)
        
        # 3. Periodic policy update
        if len(self.experience_buffer) >= self.update_frequency:
            self.update_policy()
        
        return schedule
    
    def update_policy(self):
        """
        Update policy using recent production experience.
        """
        # Sample from recent experience
        batch = sample(self.experience_buffer, batch_size=256)
        
        # Compute returns using user feedback
        for transition in batch:
            user_satisfaction = get_user_feedback(transition.schedule)
            transition.reward += user_satisfaction
        
        # Update policy
        self.policy.learn(batch)
        
        # Clear old experience (prevent catastrophic forgetting)
        self.experience_buffer = self.experience_buffer[-1000:]
```

### User Feedback Integration

```python
def get_user_feedback(schedule):
    """
    Collect user satisfaction ratings.
    """
    # Implicit feedback: schedule accepted or rejected
    if schedule.accepted_by_user:
        return +1.0
    else:
        return -1.0
    
    # Explicit feedback: user rating (1-5 stars)
    rating = schedule.user_rating
    return (rating - 3.0) / 2.0  # Normalize to [-1, 1]
```

---

## Challenge: Catastrophic Forgetting

**Problem**: Learning new tasks causes forgetting of old tasks.

**Solution: Elastic Weight Consolidation (EWC)**

```python
class EWCPolicy:
    def __init__(self, base_policy, lambda_ewc=1000):
        self.policy = base_policy
        self.lambda_ewc = lambda_ewc
        self.fisher_information = None
        self.optimal_params = None
    
    def learn_with_ewc(self, new_task_data):
        """
        Learn new task while preserving old knowledge.
        """
        # Standard RL loss
        rl_loss = compute_rl_loss(new_task_data)
        
        # EWC penalty: discourage changing important parameters
        if self.fisher_information is not None:
            ewc_loss = 0
            for param_name, param in self.policy.named_parameters():
                fisher = self.fisher_information[param_name]
                optimal = self.optimal_params[param_name]
                
                # Penalize changes to important parameters
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
            
            total_loss = rl_loss + self.lambda_ewc * ewc_loss
        else:
            total_loss = rl_loss
        
        # Update policy
        total_loss.backward()
        optimizer.step()
    
    def consolidate_knowledge(self):
        """
        Compute Fisher information after learning a task.
        """
        self.fisher_information = compute_fisher_information(self.policy)
        self.optimal_params = {
            name: param.clone() 
            for name, param in self.policy.named_parameters()
        }
```

---

## Expected Benefits

### 1. Continuous Improvement
- **Current**: Static policy after deployment
- **Expected**: Improves with every production run
- **Impact**: 10-20% improvement over 6 months

### 2. Adaptation to User Preferences
- **Current**: Hardcoded preferences (minimize gaps, etc.)
- **Expected**: Learn actual user preferences from feedback
- **Impact**: 30% higher user satisfaction

### 3. Handling Distribution Shift
- **Current**: Fails on novel problem types
- **Expected**: Adapts to new patterns automatically
- **Impact**: Robust to changing requirements

---

## Risks and Mitigations

### Risk 1: Policy Degradation
**Mitigation**: Maintain validation set, rollback if performance drops

### Risk 2: Catastrophic Forgetting
**Mitigation**: Use EWC or replay buffer with old experiences

### Risk 3: Adversarial Feedback
**Mitigation**: Filtering and validation of user feedback

---

## Implementation Roadmap

1. **Month 1**: Experience collection infrastructure
2. **Month 2**: Implement EWC for continual learning
3. **Month 3**: User feedback integration system
4. **Month 4**: Deployment and monitoring

**Difficulty**: High | **Priority**: Low (future production enhancement)
