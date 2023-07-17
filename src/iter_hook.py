
class NormalIterHook:
    def run_iter(self, trainer, X, y):
        pred = trainer.model(X)
        loss = trainer.loss_fn(pred, y)
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        return loss
    

class SamIterHook:
    def run_iter(self, trainer, X, y):
        loss = trainer.loss(trainer.model(X), y)  # use this loss for any training statistics
        loss.backward()
        trainer.optimizer.first_step(zero_grad=True)

        # second forward-backward pass
        trainer.loss(trainer.model(X), y).backward()  # make sure to do a full forward pass
        trainer.optimizer.second_step(zero_grad=True)
        return loss