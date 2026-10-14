# In neuralbench-repo/neuralbench/pl_module.py, replace BrainModule.training_step with:


def training_step(self, batch: Batch, batch_idx: int):
    loss, _, _ = self._run_step(batch, step_name="train", batch_idx=batch_idx)
    # Optional model-side training losses (e.g. UlhmPose's pose branch).
    auxiliary_losses = getattr(self.model, "auxiliary_losses", None)
    if auxiliary_losses is not None:
        target = batch.data["target"]
        for name, value in auxiliary_losses(target).items():
            self.log(
                f"train/loss_{name}",
                value,
                on_step=True,
                on_epoch=True,
                batch_size=target.shape[0],
            )
            loss = loss + value
    return loss
