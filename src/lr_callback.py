from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class LRCallback(TrainerCallback):
    """
    Learning rate callback that decays the learning rate at a specific step.
    Used for releasing frozen parameters during training.
    """
    def __init__(self, release_step: int = 2000, decay_weight: float = 0.1):
        self.release_step = release_step
        self.decay_weight = decay_weight
        self.initial_lr = None

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Store initial learning rate on first step
        if self.initial_lr is None:
            self.initial_lr = args.learning_rate

        # Decay learning rate at release_step
        if state.global_step == self.release_step:
            new_lr = self.initial_lr * self.decay_weight
            # Update the learning rate in the optimizer
            if hasattr(args, 'learning_rate'):
                # Note: This is a simplified version. In practice, you might need to
                # access the optimizer directly to update the learning rate.
                # The actual LR update might need to be done through the optimizer's
                # parameter groups, but this provides the basic structure.
                pass
        
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Decay learning rate at release_step
        if state.global_step == self.release_step:
            # Get the optimizer from the model
            model = kwargs.get('model')
            if model is not None:
                optimizer = kwargs.get('optimizer')
                if optimizer is not None:
                    # Decay learning rate for all parameter groups
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * self.decay_weight
        
        return control

