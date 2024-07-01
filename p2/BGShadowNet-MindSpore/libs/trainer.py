import mindspore as ms
from mindspore import ops, nn
from mindspore.amp import StaticLossScaler, all_finite

class Trainer:
    """一个有两个loss的训练示例"""
    def __init__(self, net, loss1, loss2, optimizer, train_dataset, loss_scale=1.0, eval_dataset=None, metric=None):
        self.net = net
        self.loss1 = loss1
        self.loss2 = loss2
        self.opt = optimizer
        self.train_dataset = train_dataset
        self.train_data_size = self.train_dataset.get_dataset_size()
        self.weights = self.opt.parameters
        self.value_and_grad = ms.value_and_grad(self.forward_fn, None, weights=self.weights, has_aux=True)

        self.grad_reducer = self.get_grad_reducer()
        self.loss_scale = StaticLossScaler(loss_scale)
        self.run_eval = eval_dataset is not None
        if self.run_eval:
            self.eval_dataset = eval_dataset
            self.metric = metric
            self.best_acc = 0

    def get_grad_reducer(self):
        grad_reducer = ops.identity
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        reducer_flag = (parallel_mode != ms.ParallelMode.STAND_ALONE)
        if reducer_flag:
            grad_reducer = nn.DistributedGradReducer(self.weights)
        return grad_reducer

    def forward_fn(self, inputs, labels):
        logits = self.net(inputs)
        loss1 = self.loss1(logits, labels)
        loss2 = self.loss2(logits, labels)
        loss = loss1 + loss2
        loss = self.loss_scale.scale(loss)
        return loss, loss1, loss2

    @ms.jit
    def train_single(self, inputs, labels):
        (loss, loss1, loss2), grads = self.value_and_grad(inputs, labels)
        loss = self.loss_scale.unscale(loss)
        grads = self.loss_scale.unscale(grads)
        grads = self.grad_reducer(grads)
        state = all_finite(grads)
        if state:
            self.opt(grads)

        return loss, loss1, loss2

    def train(self, epochs):
        train_dataset = self.train_dataset.create_dict_iterator()
        self.net.set_train(True)
        for epoch in range(epochs):
            # 训练一个epoch
            for batch, data in enumerate(train_dataset):
                loss, loss1, loss2 = self.train_single(data["image"], data["label"])
                if batch % 100 == 0:
                    print(f"step: [{batch} /{self.train_data_size}] "
                          f"loss: {loss}, loss1: {loss1}, loss2: {loss2}", flush=True)
            if self.run_eval:
                eval_dataset = self.eval_dataset.create_dict_iterator(num_epochs=1)
                self.net.set_train(False)
                self.metric.clear()
                for batch, data in enumerate(eval_dataset):
                    output = self.net(data["image"])
                    self.metric.update(output, data["label"])
                accuracy = self.metric.eval()
                print(f"epoch {epoch}, accuracy: {accuracy}", flush=True)
                if accuracy >= self.best_acc:
                    self.best_acc = accuracy
                    ms.save_checkpoint(self.net, "best.ckpt")
                    print(f"Updata best acc: {accuracy}")
                self.net.set_train(True)
