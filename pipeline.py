import torch
from time import time
import SimpleITK as sitk
import nnunetv2
from numpy import NaN
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from typing import Tuple, Union, Optional, List, Dict
from abc import ABC, abstractmethod
from torch._dynamo import OptimizedModule
from cupyx.scipy import ndimage
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from nnunetv2.architecture.repvgg_unet import plain_unet_S5, plain_unet_S4, plain_unet_702, plain_unet
from nnunetv2.preprocessing.resampling.default_resampling import fast_resample_logit_to_shape
from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.utilities.utils import log_runtime
from modelopt.torch.quantization.utils import export_torch_mode
from tqdm import tqdm
import torch_tensorrt as torchtrt
import argparse
import glob
import cupy as cp
import os
import gc

from batchgenerators.utilities.file_and_folder_operations import load_json, join

results: dict[str, float] = {}

CT_configuration = {
    "transpose_forward": [
        0,
        1,
        2
    ],
    "spacing": [
        2.5,
        0.7958984971046448,
        0.7958984971046448
    ],
    'intensity_prop': {
        "max": 3071.0,
        "mean": 97.29716491699219,
        "median": 118.0,
        "min": -1024.0,
        "percentile_00_5": -958.0,
        "percentile_99_5": 270.0,
        "std": 137.8484649658203
    }
}


class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(self, use_mask_for_norm: Optional[bool] = None, intensityproperties: Optional[Dict] = None,
                 target_dtype: torch.dtype = torch.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict) or intensityproperties is None
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: torch.Tensor, seg: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: torch.Tensor, seg: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']

        image = image.to(dtype=self.target_dtype)
        image = torch.clamp(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        return image


def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    filled_mask = ndimage.binary_fill_holes(nonzero_mask)
    return filled_mask


def get_bbox_from_mask(mask: cp.ndarray) -> List[List[int]]:
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics

    This implementation uses CuPy for GPU acceleration. The mask should be a CuPy array.

    Args:
        mask (cp.ndarray): 3D mask array on GPU

    Returns:
        List[List[int]]: Bounding box coordinates as [[minz, maxz], [minx, maxx], [miny, maxy]]
    """
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y

    # Create range arrays on GPU
    zidx = cp.arange(Z)
    xidx = cp.arange(X)
    yidx = cp.arange(Y)

    # Z dimension
    for z in zidx.get():  # .get() to iterate over CPU array
        if cp.any(mask[z]).get():  # .get() to get boolean result to CPU
            minzidx = z
            break
    for z in zidx[::-1].get():
        if cp.any(mask[z]).get():
            maxzidx = z + 1
            break

    # X dimension
    for x in xidx.get():
        if cp.any(mask[:, x]).get():
            minxidx = x
            break
    for x in xidx[::-1].get():
        if cp.any(mask[:, x]).get():
            maxxidx = x + 1
            break

    # Y dimension
    for y in yidx.get():
        if cp.any(mask[:, :, y]).get():
            minyidx = y
            break
    for y in yidx[::-1].get():
        if cp.any(mask[:, :, y]).get():
            maxyidx = y + 1
            break

    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def bounding_box_to_slice(bounding_box: List[List[int]]):
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    https://chatgpt.com/share/679203ec-3fbc-8013-a003-13a7adfb1e73
    """
    return tuple([slice(*i) for i in bounding_box])


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]

    slicer = (slice(None),) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox


def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape


def fast_resample_data_or_seg_to_shape(data: Union[torch.Tensor, np.ndarray],
                                       new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                       current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                       new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                       is_seg: bool = False,
                                       order: int = 3, order_z: int = 0,
                                       force_separate_z: Union[bool, None] = False,
                                       separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    use_gpu = True
    device = torch.device("cuda" if use_gpu else "cpu")
    order_to_mode_map = {
        0: "nearest",
        1: "trilinear" if new_shape[0] > 1 else "bilinear",
        2: "trilinear" if new_shape[0] > 1 else "bilinear",
        3: "trilinear" if new_shape[0] > 1 else "bicubic",
        4: "trilinear" if new_shape[0] > 1 else "bicubic",
        5: "trilinear" if new_shape[0] > 1 else "bicubic",
    }
    resize_fn = torch.nn.functional.interpolate
    kwargs = {
        'mode': order_to_mode_map[order],
        'align_corners': False
    }
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        if not isinstance(data, torch.Tensor):
            #torch_data = torch.from_numpy(data).float()
            torch_data = torch.as_tensor(data.get())
        else:
            torch_data = data.float()
        if new_shape[0] == 1:
            torch_data = torch_data.transpose(1, 0)
            new_shape = new_shape[1:]
        else:
            torch_data = torch_data.unsqueeze(0)

        torch_data = resize_fn(torch_data.to(device), tuple(new_shape), **kwargs)

        if new_shape[0] == 1:
            torch_data = torch_data.transpose(1, 0)
        else:
            torch_data = torch_data.squeeze(0)

        # if use_gpu:
        #     torch_data = torch_data.cpu()
        reshaped_final_data = torch_data
        # if isinstance(data, np.ndarray):
        #     reshaped_final_data = torch_data.numpy().astype(dtype_data)
        # else:
        #     reshaped_final_data = torch_data.to(dtype_data)

        #print(f"Reshaped data from {shape} to {new_shape}")
        #print(f"reshaped_final_data shape: {reshaped_final_data.shape}")
        assert reshaped_final_data.ndim == 4, f"reshaped_final_data.shape = {reshaped_final_data.shape}"
        return reshaped_final_data
    else:
        print("no resampling necessary")
        return data


@log_runtime
def logits_to_segmentation(predicted_logits):
    max_logit, max_class = torch.max(predicted_logits, dim=0)

    # Apply threshold: Only assign the class if its logit exceeds the threshold
    segmentation = torch.where(max_logit >= 0.5, max_class, torch.tensor(0, device=predicted_logits.device))
    return segmentation


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                use_softmax,
                                                                return_probabilities: bool = False,
                                                                ):
    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]

    # apply_inference_nonlin will convert to torch
    if properties_dict['shape_after_cropping_and_before_resampling'][0] < 600:
        predicted_logits = fast_resample_logit_to_shape(predicted_logits,
                                                        properties_dict['shape_after_cropping_and_before_resampling'],
                                                        current_spacing,
                                                        [properties_dict['spacing'][i] for i in
                                                         plans_manager.transpose_forward])
        gc.collect()
        empty_cache(predicted_logits.device)
        if use_softmax:
            predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)

            del predicted_logits

            # Start timing for converting probabilities to segmentation
            segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
        else:
            # Get the class with the maximum logit at each pixel
            segmentation = logits_to_segmentation(predicted_logits)

    else:

        segmentation = fast_resample_logit_to_shape(predicted_logits,
                                                    properties_dict['shape_after_cropping_and_before_resampling'],
                                                    current_spacing,
                                                    [properties_dict['spacing'][i] for i in
                                                     plans_manager.transpose_forward])

    dtype = torch.uint8 if len(label_manager.foreground_labels) < 255 else torch.uint16
    segmentation_reverted_cropping = torch.zeros(properties_dict['shape_before_cropping'], dtype=dtype)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation

    del segmentation

    # Revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.permute(plans_manager.transpose_backward)

    return segmentation_reverted_cropping.cpu()


class SimplePredictor(nnUNetPredictor):
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None
            ckpt = checkpoint['network_weights']
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            parameters.append(ckpt)

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')

        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        if 'S4' in model_training_output_dir:
            network = plain_unet_S4(14, False, False)
        elif 'S5' in model_training_output_dir:
            network = plain_unet_S5(14, False, False)
        else:
            network = trainer_class.build_network_architecture(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                num_input_channels,
                plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
                enable_deep_supervision=False
            )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])
        for params in self.list_of_parameters:
            self.network.load_state_dict(params)

        for module in self.network.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def preprocess(self, image, properties):
        preprocessor = self.configuration_manager.preprocessor_class(verbose=False)
        image = torch.from_numpy(image).to(dtype=torch.float32, memory_format=torch.contiguous_format).to(self.device)
        # data = preprocessor.run_case_npy(image, None, props, self.plans_manager, self.configuration_manager, self.dataset_json)
        data = image.clone()
        data = data.permute([0, *[i + 1 for i in CT_configuration['transpose_forward']]])
        original_spacing = [properties['spacing'][i] for i in CT_configuration['transpose_forward']]
        t0 = time()
        data, seg, bbox = crop_to_nonzero(data)
        results["cropping"].append(time() - t0)
        torch.cuda.synchronize()
        target_spacing = CT_configuration['spacing']
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
        normalization = CTNormalization(intensityproperties=CT_configuration['intensity_prop'])
        t0 = time()
        data = normalization.run(data.cuda())
        torch.cuda.synchronize()
        results["normalization"].append(time() - t0)
        fast_resample_data_or_seg_to_shape(data, new_shape, original_spacing, target_spacing)
        torch.cuda.synchronize()
        results["resampling"].append(time() - t0)
        return data

    @log_runtime
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)
                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    def inference(self, image, properties_dict, use_softmax):
        image = self.preprocess(image, properties_dict)

        with torch.no_grad():
            assert isinstance(image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()
            empty_cache(self.device)

            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                data, slicer_revert_padding = pad_nd_image(image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                       self.perform_everything_on_device)

                empty_cache(self.device)  # Start time for inference time calculation
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

                segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits,
                                                                                           self.plans_manager,
                                                                                           self.configuration_manager,
                                                                                           self.label_manager,
                                                                                           properties_dict,
                                                                                           use_softmax,
                                                                                           return_probabilities=False,
                                                                                           )

        return segmentation


def compute_dice_coefficient(pred: torch.Tensor, gt: torch.Tensor) -> float:
    volume_sum = gt.sum() + pred.sum()
    if volume_sum == 0:
        return NaN
    volume_intersect = (gt & pred).sum()
    return 2 * volume_intersect / volume_sum


if __name__ == '__main__':
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Inference for nnUNet model")
        parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to the input image file')
        parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the output segmentation')
        parser.add_argument('--model_path', type=str, required=True, help='Name of the model to use for inference')
        parser.add_argument('--fold', type=str, default='all', help='Fold number to use for inference (default: 0)')
        parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth',
                            help='Path to the model checkpoint file')
        parser.add_argument('--use_softmax', default=False, help='Apply softmax to the output probabilities')
        parser.add_argument('--trt', action='store_true', help='Using TensorRT')
        parser.add_argument('--onnx_trt', action='store_true', help='Using TensorRT')
        parser.add_argument('--run_engine_trt', action='store_true', help='Using TensorRT')
        parser.add_argument('--bn_trt', action='store_true', help='Using TensorRT')

        return parser.parse_args()


    args = parse_arguments()

    device = torch.device('cuda', 0)
    predictor = SimplePredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    predictor.initialize_from_trained_model_folder(
        args.model_path,
        use_folds=args.fold,
        checkpoint_name=args.checkpoint,
    )
    predictor.network.to(device)

    input_folder = args.input_path
    output_folder = args.output_path
    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, '*'))

    if args.bn_trt:
        input_shape = (1, 1, 64, 256, 256)
        model = predictor.network
        model.cuda()
        model.eval()

        data = torch.randn(input_shape).to("cuda")
        with torch.no_grad():
            with export_torch_mode():
                # Compile the model with Torch-TensorRT Dynamo backend
                # input_tensor = images.cuda()
                input_tensor = torch.randn(input_shape).to("cuda")
                # torch.export.export() failed due to RuntimeError: Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()
                from torch.export._trace import _export

                exp_program = _export(model, (input_tensor,))
                # exp_program = torchtrt.dynamo.export(model, (input_tensor,))
                # enabled_precisions = {torch.float}
                enabled_precisions = {torch.half}  # , torch.int8 torch.half,
                trt_model = torchtrt.dynamo.compile(
                    exp_program,
                    inputs=[input_tensor],
                    enabled_precisions=enabled_precisions,
                    min_block_size=1,
                )

                predictor.network = trt_model

    for file in tqdm(files):
        image, props = SimpleITKIO().read_images([file])
        logit = load(f"inference_test_logit/{case[:case.find('.')]}.pt".replace("_0000", "")).to("cuda")
        seg = predictor.inference(image, props, args.use_softmax)
        print(compute_dice_coefficient(seg, ))
        sitk_img = sitk.GetImageFromArray(seg)
        sitk_img.SetSpacing(props['sitk_stuff']['spacing'])
        sitk_img.SetOrigin(props['sitk_stuff']['origin'])
        sitk_img.SetDirection(props['sitk_stuff']['direction'])
        case_name = file.split('/')[-1].replace('_0000.nii.gz', '.nii.gz')
        sitk.WriteImage(sitk_img, os.path.join(output_folder, f'{case_name}'))
