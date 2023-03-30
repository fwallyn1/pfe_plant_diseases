from copy import deepcopy, copy
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, all_target_layers, reshape_transform):
        self.model = model.eval()
        self.n_views = len(all_target_layers)
        self.gradients = [[] for i in range(self.n_views)]
        self.activations = [[] for i in range(self.n_views)]
        self.reshape_transform = reshape_transform
        self.handles = []
        self.view_activations = 0
        self.view_gradients = 0
        for i,target_layers in enumerate(all_target_layers):
            for target_layer in target_layers:
                self.handles.append(
                    target_layer.register_forward_hook(self.save_activation))
                # Because of https://github.com/pytorch/pytorch/issues/61519,
                # we don't use backward hook to record gradients.
                self.handles.append(
                    target_layer.register_forward_hook(self.save_gradient))
                
    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        #print(self.view_activations)
        self.activations[self.view_activations].append(activation.cpu().detach())
        self.view_activations +=1

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients[self.view_gradients] = [grad.cpu().detach()] + self.gradients[self.view_gradients]
            self.view_gradients +=1
        output.register_hook(_store_grad)
        

    def __call__(self, x):
        self.gradients = [[] for i in range(self.n_views)]
        self.activations = [[] for i in range(self.n_views)]
        self.view_gradients = 0
        self.view_activations = 0
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
        self.view_gradients = 0
        self.view_activations = 0