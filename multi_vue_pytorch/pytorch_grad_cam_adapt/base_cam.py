import numpy as np
import torch
from typing import Callable, List, Tuple
from pytorch_grad_cam_adapt.activations_and_gradients import ActivationsAndGradients
#from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam_adapt.image import scale_cam_image
from pytorch_grad_cam_adapt.model_targets import ClassifierOutputTarget
from copy import deepcopy

class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[List[torch.nn.Module]],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, None)
        self.cuda = use_cuda
        if self.cuda:
            self.model = self.model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        """self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)"""

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam


    def forward(self,
                input_tensors: List[torch.Tensor],
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:
        
        if self.cuda:
            for input_tensor in input_tensors :
                input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            for input_tensor in input_tensors :
                input_tensor = torch.autograd.Variable(input_tensor,
                                                       requires_grad=True)
        for params in self.model.parameters():
            params.requires_grad = True                           
        outputs = self.activations_and_grads(input_tensors)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]
        
        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                       for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        all_cam_per_layer = self.compute_cam_per_layer(input_tensors,
                                                   targets,
                                                   eigen_smooth)
        for params in self.model.parameters():
            params.requires_grad = False
        #self.activations_and_grads.release()
        self.activations_and_grads.view_activations = 0
        self.activations_and_grads.view_gradients = 0
        #print("base cam",self.activations_and_grads.view_activations)
        return self.aggregate_multi_layers(all_cam_per_layer) , { 'prediction': target_categories}

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensors: List[torch.Tensor],
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = []
        for activation in self.activations_and_grads.activations:
            activations_list.append([a.cpu().data.numpy()
                               for a in activation])
        grads_list = []
        self.activations_and_grads.gradients.reverse()
        for gradient in self.activations_and_grads.gradients:
            grads_list.append([g.cpu().data.numpy()
                          for g in gradient])
        target_size = self.get_target_width_height(input_tensors[0])
        
        all_cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for j in range(len(self.target_layers)):
            cam_per_target_layer = []
            for i in range(len(self.target_layers[j])):
                target_layer = self.target_layers[j][i]
                layer_activations = None
                layer_grads = None
                if i < len(activations_list[j]):
                    layer_activations = activations_list[j][i]
                if i < len(grads_list[j]):
                    layer_grads = grads_list[j][i]
                cam = self.get_cam_image(input_tensors[j],
                                         target_layer,
                                         targets,
                                         layer_activations,
                                         layer_grads,
                                         eigen_smooth)
                cam = np.maximum(cam, 0)
                scaled = scale_cam_image(cam, target_size)
                cam_per_target_layer.append(scaled[:, None, :])
            all_cam_per_target_layer.append(cam_per_target_layer)
        return all_cam_per_target_layer

    def aggregate_multi_layers(
            self,
            all_cam_per_target_layer: List[np.ndarray]) -> List[np.ndarray]:
        all_scaled_images = []
        for cam_per_target_layer in all_cam_per_target_layer :
            cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
            cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
            result = np.mean(cam_per_target_layer, axis=1)
            all_scaled_images.append(scale_cam_image(result))
        return all_scaled_images

    """def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam"""

    def __call__(self,
                 input_tensors: List[torch.Tensor],
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)
        return self.forward(input_tensors,
                            targets, eigen_smooth)
    
    def __del__(self):
        pass
        #self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True