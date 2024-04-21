import re
from typing import Tuple
from Levenshtein import distance
import cv2
from matplotlib import pyplot as plt
import pix2tex.utils as p2t_utils
import pix2tex.models as p2t_models
import yaml
from munch import Munch
import torch
from pix2tex.dataset.dataset import Im2LatexDataset
import numpy as np
from collections import defaultdict
from pix2tex.eval import detokenize
from torchtext.data import metrics

from pix2tex.utils.utils import alternatives, post_process, token2str
from tqdm import tqdm
from pix2tex import multiline_utils
from pix2tex.cli import minmax_size
from PIL import Image

# CONFIG_PATH = "pix2tex/model/settings/config.yaml"
# BATCHSIZE = 1
# TEMPERATURE = .9
# CHECKPOINT_PATH = "hw_checkpoints/handwritten_training/handwritten_training_e19_step63.pth"
# DEVICE = "cpu"
# DATA_PATH = "pix2tex/dataset/handwritten/test.pkl"

def get_model_and_data(config_path, checkpoint_path, data_path, batch_size=1, temperature=.2,  device="cpu"):
    """
    Get the model and data for evaluation, along with configuration arguments.

    Inputs:
    config_path (str): path to the configuration file
    batch_size (int): batch size for evaluation
    temperature (float): temperature for evaluation
    checkpoint_path (str): path to the model checkpoint
    device (str): device to run the model on
    data_path (str): path to the data

    Returns:
    Tuple[torch.nn.Module, Im2LatexDataset, Munch]: model, dataset, arguments
    """

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args = p2t_utils.parse_args(Munch(config))

    args.testbatchsize = batch_size
    args.wandb = False
    args.temperature = temperature

    model = p2t_models.get_model(args)
    model.load_state_dict(torch.load(checkpoint_path, device))

    dataset = Im2LatexDataset(pad=True).load(data_path)
    valargs = args.copy()
    valargs.update(batchsize=args.testbatchsize, keep_smaller_batches=True, test=True)
    dataset.update(**valargs)

    return model, dataset, args

def extraer_numero(texto):
    """
    Función para extraer el primer número encontrado en una cadena de texto.

    Parámetros:
    - texto (str): La cadena de texto de la cual extraer el número.

    Retorna:
    - str: El primer número encontrado en la cadena de texto.
    - None: Si no se encuentra ningún número.
    """
    # Utilizar expresión regular para encontrar todos los números en la cadena
    numeros = re.findall(r'\d+', texto)

    # Asumiendo que solo hay un número en la cadena, obtener el primer resultado
    numero = numeros[0] if numeros else None

    return numeros


def evaluate(model, dataset: Im2LatexDataset, args: Munch, num_batches: int = None):
    """evaluates the model. Returns bleu score on the dataset

    Args:
        model (torch.nn.Module): the model
        dataset (Im2LatexDataset): test dataset
        args (Munch): arguments
        num_batches (int): How many batches to evaluate on. Defaults to None (all batches).
        name (str, optional): name of the test e.g. val or test for wandb. Defaults to 'test'.

    Returns:
        Tuple[float, float, float]: BLEU score of validation set, normed edit distance, token accuracy
    """
    assert len(dataset) > 0
    device = args.device
    bleus, edit_dists, token_acc = [], [], []
    bleu_score, edit_distance, token_accuracy = 0, 1, 0
    iter_ds = iter(dataset)
    pbar = tqdm(enumerate(iter_ds), total=len(dataset))
    preds = defaultdict(list)
    pred_truth = defaultdict(list)
    for i, (seq, im) in pbar:
        if seq is None or im is None:
            continue
        #loss = decoder(tgt_seq, mask=tgt_mask, context=encoded)
        dec = model.generate(im.to(device), temperature=args.get('temperature', .2))
        pred = detokenize(dec, dataset.tokenizer)
        tokenized_pred = token2str(dec, dataset.tokenizer)
        preds[i].append(pred)
        truth = detokenize(seq['input_ids'], dataset.tokenizer)
        tokenized_truth = token2str(seq['input_ids'], dataset.tokenizer)
        bleus.append(metrics.bleu_score(pred, [alternatives(x) for x in truth]))
        for predi, truthi in zip(token2str(dec, dataset.tokenizer), token2str(seq['input_ids'], dataset.tokenizer)):
            ts = post_process(truthi)
            if len(ts) > 0:
                edit_dist = distance(post_process(predi), ts)/len(ts)
                edit_dists.append(distance(post_process(predi), ts)/len(ts))
        dec = dec.cpu()
        tgt_seq = seq['input_ids'][:, 1:]
        shape_diff = dec.shape[1]-tgt_seq.shape[1]
        if shape_diff < 0:
            dec = torch.nn.functional.pad(dec, (0, -shape_diff), "constant", args.pad_token)
        elif shape_diff > 0:
            tgt_seq = torch.nn.functional.pad(tgt_seq, (0, shape_diff), "constant", args.pad_token)
        mask = torch.logical_or(tgt_seq != args.pad_token, dec != args.pad_token)
        tok_acc = (dec == tgt_seq)[mask].float().mean().item()
        token_acc.append(tok_acc)
        pbar.set_description('BLEU: %.3f, ED: %.2e, ACC: %.3f' % (np.mean(bleus), np.mean(edit_dists), np.mean(token_acc)))

        #Busco el nombre de la imagen 
        batch = iter_ds.pairs[iter_ds.i - 1]
        _,ims=batch.T
        label = extraer_numero(ims[0])[1]
        pred_truth[label] = {'predicted': tokenized_pred,
                             'truth':tokenized_truth,
                             'pred_tokens':pred,
                             'truth_tokens':truth,
                             'bleu':bleu_score,
                             'edit_dist': edit_dist,
                             'token acc':tok_acc}
        if num_batches is not None and i >= num_batches:
            break
    if len(bleus) > 0:
        bleu_score = np.mean(bleus)
    if len(edit_dists) > 0:
        edit_distance = np.mean(edit_dists)
    if len(token_acc) > 0:
        token_accuracy = np.mean(token_acc)

    print('\n%s\n%s' % (truth, pred))
    print('BLEU: %.2f' % bleu_score)
    return bleu_score, edit_distance, token_accuracy, bleus, edit_dists, token_acc, pred_truth

def parse_prediction(pred):
    pred = np.array(pred).squeeze()
    return ''.join(pred)


def resize_line(img: np.ndarray, max_dimensions: Tuple[int, int] = None, min_dimensions: Tuple[int, int] = None) -> np.ndarray:
    """Resize or pad an image to fit into given dimensions

    Args:
        img (np.ndarray): Image to scale up/down.
        max_dimensions (Tuple[int, int], optional): Maximum dimensions. Defaults to None.
        min_dimensions (Tuple[int, int], optional): Minimum dimensions. Defaults to None.

    Returns:
        np.ndarray: Image with correct dimensionality
    """
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.shape[:2][::-1], max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.shape[:2][::-1])//max(ratios)
            img = cv2.resize(img, tuple(size.astype(int)), interpolation=cv2.INTER_LINEAR)
    if min_dimensions is not None:
        # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
        padded_size = [max(img_dim, min_dim) for img_dim, min_dim in zip(img.shape[:2][::-1], min_dimensions)]
        if padded_size != list(img.shape[:2][::-1]):  # assert hypothesis
            padded_im = np.full((*padded_size[::-1], img.shape[2]), 255, dtype=np.float32) if len(img.shape) == 3 else np.full(padded_size[::-1], 255, dtype=np.float32)
            y_offset = (padded_im.shape[0] - img.shape[0]) // 2
            x_offset = (padded_im.shape[1] - img.shape[1]) // 2
            padded_im[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
            img = padded_im
    return img


def evaluate_multiline(model, dataset: Im2LatexDataset, args: Munch, candidate_sizes: list, num_batches: int = None):
    """evaluates the model. Returns bleu score on the dataset

    Args:
        model (torch.nn.Module): the model
        dataset (Im2LatexDataset): test dataset
        args (Munch): arguments
        num_batches (int): How many batches to evaluate on. Defaults to None (all batches).
        name (str, optional): name of the test e.g. val or test for wandb. Defaults to 'test'.

    Returns:
        Tuple[float, float, float]: BLEU score of validation set, normed edit distance, token accuracy
    """
    assert len(dataset) > 0
    device = args.device
    bleus, edit_dists, token_acc = [], [], []
    bleu_score, edit_distance, token_accuracy = 0, 1, 0
    iter_ds = iter(dataset)
    pbar = tqdm(enumerate(iter_ds), total=len(dataset))
    preds = defaultdict(list)
    pred_truth = defaultdict(list)
    for i, (seq, im) in pbar:
        if seq is None or im is None:
            continue
        
        pre_split_img = multiline_utils.ImageTensor(im.squeeze(0), th=0, candidate_sizes=candidate_sizes)
        lines = pre_split_img.split_img_into_lines()
        print("detected lines --> ", len(lines))

        # compute inference over multiple lines
        # pred = []
        running_line_pred = []
        for line in lines:
            # to_pred = resize_line(line, max_dimensions=dataset.max_dimensions, min_dimensions=dataset.min_dimensions)
            to_pred = line.unsqueeze(0)

            dec = model.generate(to_pred.to(device), temperature=args.get('temperature', .2))
            detok_dec = detokenize(dec, dataset.tokenizer)
            print("Len detok_dec --> ", len(detok_dec))
            print("Len detok_dec @ index 0--> ", len(detok_dec[0]))
            running_line_pred.extend(detok_dec[0])
            print("Len runnig pred --> ", len(running_line_pred))

        pred = [running_line_pred]


        tokenized_pred = None
        preds[i].append(pred)
        truth = detokenize(seq['input_ids'], dataset.tokenizer)
        print("Len of truth --> ", len(truth))
        tokenized_truth = token2str(seq['input_ids'], dataset.tokenizer)
        bleus.append(metrics.bleu_score(pred, [alternatives(x) for x in truth]))
        for predi, truthi in zip(token2str(dec, dataset.tokenizer), token2str(seq['input_ids'], dataset.tokenizer)):
            ts = post_process(truthi)
            if len(ts) > 0:
                edit_dist = distance(post_process(predi), ts)/len(ts)
                edit_dists.append(distance(post_process(predi), ts)/len(ts))
        dec = dec.cpu()
        tgt_seq = seq['input_ids'][:, 1:]
        shape_diff = dec.shape[1]-tgt_seq.shape[1]
        if shape_diff < 0:
            dec = torch.nn.functional.pad(dec, (0, -shape_diff), "constant", args.pad_token)
        elif shape_diff > 0:
            tgt_seq = torch.nn.functional.pad(tgt_seq, (0, shape_diff), "constant", args.pad_token)
        mask = torch.logical_or(tgt_seq != args.pad_token, dec != args.pad_token)
        tok_acc = (dec == tgt_seq)[mask].float().mean().item()
        token_acc.append(tok_acc)
        pbar.set_description('BLEU: %.3f, ED: %.2e, ACC: %.3f' % (np.mean(bleus), np.mean(edit_dists), np.mean(token_acc)))

        #Busco el nombre de la imagen 
        batch = iter_ds.pairs[iter_ds.i - 1]
        _,ims=batch.T
        label = extraer_numero(ims[0])[1]
        pred_truth[label] = {'predicted': tokenized_pred,
                             'truth':tokenized_truth,
                             'pred_tokens':pred,
                             'truth_tokens':truth,
                             'bleu':bleu_score,
                             'edit_dist': edit_dist,
                             'token acc':tok_acc}
        if num_batches is not None and i >= num_batches:
            break
    if len(bleus) > 0:
        bleu_score = np.mean(bleus)
    if len(edit_dists) > 0:
        edit_distance = np.mean(edit_dists)
    if len(token_acc) > 0:
        token_accuracy = np.mean(token_acc)

    print('\n%s\n%s' % (truth, pred))
    print('BLEU: %.2f' % bleu_score)
    return bleu_score, edit_distance, token_accuracy, bleus, edit_dists, token_acc, pred_truth

