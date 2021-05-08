import warnings
warnings.filterwarnings('ignore')
from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import time
import random
from fractions import Fraction
import json

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
                       english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in generate_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["["], word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["["] or decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["]"]:
                res += [word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["EOS"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"],
                                      word2index["*"], word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["["], word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"],
                        word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums +\
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str


def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # pdb.set_trace()
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index.bool(), 0.0)


def train_attn(input_batch, input_length, target_batch, target_length, num_batch, nums_stack_batch, copy_nums,
               generate_nums, encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang, clip=0,
               use_teacher_forcing=1, beam_size=1, english=False):
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_start = output_lang.n_words - copy_nums - 2
    unk = output_lang.word2index["UNK"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)

    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_length, None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)

    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk)
            target[t] = decoder_input
    else:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                                               num_start, copy_nums, generate_nums, english)
                if USE_CUDA:
                    rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output
            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], nums_stack_batch, num_start, unk)
    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()

    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    )

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    if clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return return_loss


def evaluate_attn(input_seq, input_length, num_list, copy_nums, generate_nums, encoder, decoder, output_lang,
                  beam_size=1, english=False, max_length=MAX_OUTPUT_LENGTH):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 2

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(1)
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, [input_length], None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]])  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    beam_list = list()
    score = 0
    beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

    # Run through decoder
    for di in range(max_length):
        temp_list = list()
        beam_len = len(beam_list)
        for xb in beam_list:
            if int(xb.input_var[0]) == output_lang.word2index["EOS"]:
                temp_list.append(xb)
                beam_len -= 1
        if beam_len == 0:
            return beam_list[0].all_output
        beam_scores = torch.zeros(decoder.output_size * beam_len)
        hidden_size_0 = decoder_hidden.size(0)
        hidden_size_2 = decoder_hidden.size(2)
        all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
        if USE_CUDA:
            beam_scores = beam_scores.cuda()
            all_hidden = all_hidden.cuda()
        all_outputs = []
        current_idx = -1

        for b_idx in range(len(beam_list)):
            decoder_input = beam_list[b_idx].input_var
            if int(decoder_input[0]) == output_lang.word2index["EOS"]:
                continue
            current_idx += 1
            decoder_hidden = beam_list[b_idx].hidden

            # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
            #                                1, num_start, copy_nums, generate_nums, english)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
            score = f.log_softmax(decoder_output, dim=1)
            score += beam_list[b_idx].score
            beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
            all_hidden[current_idx] = decoder_hidden
            all_outputs.append(beam_list[b_idx].all_output)
        topv, topi = beam_scores.topk(beam_size)

        for k in range(beam_size):
            word_n = int(topi[k])
            word_input = word_n % decoder.output_size
            temp_input = torch.LongTensor([word_input])
            indices = int(word_n / decoder.output_size)

            temp_hidden = all_hidden[indices]
            temp_output = all_outputs[indices]+[word_input]
            temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0].all_output


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()

class Opt_Result:
    def __init__(self, optorA, optorB):
        self.NUM_opt_count = {}    # 用来统计每个数字已经运算了几次
        self.NUM_count = {}  # 用来统计每个数字在公式中出现了几次
        self.dist = {}

        if isinstance(optorA,str) and isinstance(optorB,str):
            self.NUM_opt_count[optorA] = 1
            self.NUM_count[optorA] = 1

            if optorB in self.NUM_count:
                temp = optorB
                optorB = optorB+'__'+str(self.NUM_count[optorB])
                self.NUM_opt_count[optorB] = 1
                self.NUM_count[temp] += 1
            else:
                self.NUM_opt_count[optorB] = 1
                self.NUM_count[optorB] = 1 
            # 这里可以检测一下dist[optorA][optorB]现在不存在
            if optorA not in self.dist:
                self.dist[optorA]={}
            if optorB not in self.dist:
                self.dist[optorB]={}
            self.dist[optorA][optorB] = 2
            self.dist[optorB][optorA] = 2

        elif  isinstance(optorA,str) and isinstance(optorB,Opt_Result):
            # 直接先融合 Opt_Result B的内容
            self.dist = optorB.dist.copy()
            optorB.step()
            self.NUM_opt_count = optorB.NUM_opt_count.copy()
            self.NUM_count = optorB.NUM_count.copy()

            if optorA in self.NUM_count:
                temp = optorA
                optorA = optorA+'__'+str(self.NUM_count[optorA])
                self.NUM_count[temp] += 1
            else:
                self.NUM_count[optorA] = 1
            self.NUM_opt_count[optorA] = 1

            
            for k,v in optorB.NUM_opt_count.items():
                if optorA not in self.dist:
                    self.dist[optorA] = {}
                # TODO comment this; debug
                if k not in self.dist:
                    self.dist[k] = {}
                try:
                    self.dist[optorA][k] = v+1
                    self.dist[k][optorA] = v+1
                except:
                    pdb.set_trace()
            
            
        elif isinstance(optorA,Opt_Result) and isinstance(optorB,str):
            self.dist = optorA.dist.copy()
            optorA.step()
            self.NUM_opt_count = optorA.NUM_opt_count.copy()
            self.NUM_count = optorA.NUM_count.copy()

            if optorB in self.NUM_count:
                temp = optorB
                optorB = optorB+'__'+str(self.NUM_count[optorB])
                self.NUM_count[temp] += 1
            else:
                self.NUM_count[optorB] = 1 
            self.NUM_opt_count[optorB] = 1
            
            # print('before')
            # print(self.dist)
            for k,v in optorA.NUM_opt_count.items():
                if optorB not in self.dist:
                    self.dist[optorB] = {}
                # TODO
                if k not in self.dist:
                    self.dist[k] = {}
                self.dist[optorB][k] = v+1
                self.dist[k][optorB] = v+1
            # print('after')
            # print(self.dist)
            # pdb.set_trace()
            

        elif isinstance(optorA,Opt_Result) and isinstance(optorB,Opt_Result):
            self.dist = optorA.dist.copy()
            optorA.step()
            optorB.step()
            self.NUM_opt_count = optorA.NUM_opt_count.copy()
            self.NUM_count = optorA.NUM_count.copy()

            temp_optorB_NUM_opt_count = optorB.NUM_opt_count.copy()

            for k,v in optorB.NUM_opt_count.items():
                pre_k = k
                short_k = k.split('__')[0]
                if short_k in self.NUM_count:
                    k = short_k+'__'+str(self.NUM_count[short_k]) 
                    self.NUM_count[short_k] += 1
                    # 将optorB中dist的旧变量转换成新的变量名，然后存在临时temp b中
                    try:
                        temp_optorB_NUM_opt_count[k] = temp_optorB_NUM_opt_count.pop(pre_k)
                    except:
                        pdb.set_trace()
                    try:
                        optorB.dist[k] = {}
                        optorB.dist[k] = optorB.dist.pop(pre_k)
                    except:
                        pdb.set_trace()
                    for k1,v1 in optorB.dist.items():
                        if pre_k in v1:
                            optorB.dist[k1][k] = optorB.dist[k1].pop(pre_k)
                    self.NUM_opt_count[k] = v
                    
                else:
                    self.NUM_opt_count[short_k] = v
                    self.NUM_count[short_k] = 1

                # 将optorB的距离记录，完全复制到新的self.dist
            for k,v in temp_optorB_NUM_opt_count.items():
                for k2,v2 in optorB.dist[k].items():
                    if k not in self.dist:
                        self.dist[k] = {}
                    self.dist[k][k2] = v2
                    if k2 not in self.dist:
                        self.dist[k2] = {}
                    self.dist[k2][k] = v2

            # 两个optor_result A B 之间的点的距离计算；
            for k,v in temp_optorB_NUM_opt_count.items():
                for k_A,v_A in optorA.NUM_opt_count.items():
                    # TODO
                    if k_A not in self.dist:
                        self.dist[k_A] = {}
                    if k not in self.dist:
                        self.dist[k] = {}
                    self.dist[k_A][k] = v+v_A
                    self.dist[k][k_A] = v+v_A      

    def step(self):
        for k in self.NUM_opt_count:
            self.NUM_opt_count[k] += 1


class Stack(list):
    def __init__(self,output_lang):
        self.base = []
        # self.dist = {}
        self.output_lang = output_lang
        # self.opt_result
    
    def push(self,c):
        self.base.append(c)
        self.update()

    def update(self):
        # print(self.base)
        # print()
        # (self.base[-3] in self.output_lang.index2word[5:22] and isinstance(self.base[-3],str)) 
        if len(self.base) >=3 and  (self.base[-1] in self.output_lang.index2word[5:22] if isinstance(self.base[-1],str) else isinstance(self.base[-1],Opt_Result)) \
            and (self.base[-2] in self.output_lang.index2word[5:22] if isinstance(self.base[-2],str) else isinstance(self.base[-2],Opt_Result)) \
            and (self.base[-3] in self.output_lang.index2word[0:5] and isinstance(self.base[-3],str)):
            right = self.base.pop()
            left = self.base.pop()
            opt = self.base.pop()
            temp_opt_result = Opt_Result(left, right) # 这一步应该计算哪两个新的元素见面了
            self.push(temp_opt_result)


def NUM_to_float(num):
    num = num.replace(')','').replace('(','')
    if num.find('/') != -1:
        return float(Fraction(num))
    elif num.find('%') != -1:
        num = num.replace('%','')
        return float(num)/100.
    else:
        return float(num)

#  先写一个function，可以将该equation中的任何2个num之间的距离求出来；
#  这是一个数学题级别的计算；不是成batch的
def compute_tree_distance(idx_equation, lang):
    equation = equation_from_index(idx_equation,lang)
    # equation = ['/', '-', 'N3', 'N1', '*', '/', 'N1', 'N0', '/', 'N2', '+', 'N2', 'N3']
    stack = Stack(lang)
    # print(equation)
    # pdb.set_trace()
    Num_list = []
    for c in equation:
        if c in lang.index2word[7:-1] and c not in Num_list:
            Num_list.append(c)
        stack.push(c)
    # print(equation)
    return (stack,equation, Num_list)
    # print(stack.base[0].dist)
    # pdb.set_trace()

def number_n_gram_word(pairs, n_gram=3):
    word_number_list = {}   # key 应该是词语，value是一个list，包含了经历过的num
    for p1, p2, p3, p4 in pairs:
        for idx in range(len(p4)):
            num_pos = p4[idx]
            # num = NUM_to_float(p3[idx])
            num = p3[idx]
            # if num not in word_number_list:
            #     word_number_list[num] = []
            
            for i in range(num_pos-n_gram,num_pos+n_gram+1):
                if i == num_pos:
                    continue
                if i >= len(p1) or i < 0:
                    continue
                if p1[i] not in word_number_list:
                    word_number_list[p1[i]] = []
                word_number_list[p1[i]].append(num)
    
    # 统计每个词语附近num的数量，num的平均值，num的类型分布；
    word_value_dict = {}       # 这个dict，key是关键词，value是一个dict，
                               # 里面这个dict中，key是统计的类型，value是对应的大小； 
    for k,v_list in word_number_list.items():
        if k not in word_value_dict:
            word_value_dict[k] = {'len':len(v_list)}
        total_count = 0.
        fraction_count = 0.
        percentage_count = 0.
        float_count = 0.
        int_count = 0.
        for v in v_list:
            v = v.replace(')','').replace('(','')
            if v.find('/') != -1:
                fraction_count += 1
                total_count += float(Fraction(v))
            elif v.find('%') != -1:
                percentage_count += 1
                v = v.replace('%','')
                total_count += float(v)/100.
            elif v.find('.') != -1:
                float_count += 1
                total_count += float(v)
            else:
                int_count += 1
                total_count += float(v)
        word_value_dict[k]['avg'] = total_count /  word_value_dict[k]['len']
        word_value_dict[k]['frac'] = fraction_count / word_value_dict[k]['len']
        word_value_dict[k]['perc'] = percentage_count / word_value_dict[k]['len']
        word_value_dict[k]['float'] = float_count / word_value_dict[k]['len']
        word_value_dict[k]['int'] = int_count / word_value_dict[k]['len']
    for k,v in word_value_dict.items():
        if v['len'] >= 10:
            print(k,v)
    with open('dict.json','a') as f:
        json.dump(word_value_dict,f,ensure_ascii=False)
    pdb.set_trace()
    return 
        

def train_probing_type_bert(input_batch, input_length,output_batch, output_length, encoder, probing_type_module, probing_type_optim,
               nums_batch, num_pos,output_lang):

    # print(nums_batch)
    pdb.set_trace()
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    encoder.eval()
    probing_type_module.train()
    if USE_CUDA:
        input_var = input_var.cuda()
    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs =  encoder_outputs.detach()


    # 进行数字的类别分类；float；int;分数；百分数；

    # 需要每一个样本跑一次；
    # 如果用来进行train encoder的话，每个样本单独forward会不会有影响呢？
    # 没有影响的，因为encoder的数据是每个batch一起来的；
    loss_batch = []
    correct_list_batch = []
    criterion = torch.nn.CrossEntropyLoss()
    for idx in range(len(input_batch)):

  
        for idx_np in range(len(num_pos[idx])):
            num_p = num_pos[idx][idx_np]
            # print(nums_batch[idx])

            if '%' in nums_batch[idx][idx_np]:
                target = torch.tensor(0, device='cuda').unsqueeze(dim=0)
            elif '/' in nums_batch[idx][idx_np]:
                target = torch.tensor(1, device='cuda').unsqueeze(dim=0)
            elif '.' in nums_batch[idx][idx_np]:
                target = torch.tensor(2, device='cuda').unsqueeze(dim=0)
            else:
                target = torch.tensor(3, device='cuda').unsqueeze(dim=0)
            # if target.item() > 10. or target.item() < -10:
            #     continue
            input_x = encoder_outputs[num_p][idx].unsqueeze(dim=0)

            pred = probing_type_module(input_x)
            # pdb.set_trace()
            loss_np = criterion(pred,target)
            correct_list_batch.append((torch.max(pred,1)[1]==target).item())
            loss_batch.append(loss_np)

        
    loss = sum(loss_batch) / len(loss_batch)

    probing_type_optim.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(probing_regression_module.parameters(),clip_value=1.1)
    probing_type_optim.step()
    return loss.item(),correct_list_batch

def train_probing_type(input_batch, input_length,output_batch, output_length, encoder, probing_type_module, probing_type_optim,
               nums_batch, num_pos,output_lang):

    # print(nums_batch)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    encoder.eval()
    probing_type_module.train()
    if USE_CUDA:
        input_var = input_var.cuda()
    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs =  encoder_outputs.detach()


    # 进行数字的类别分类；float；int;分数；百分数；

    # 需要每一个样本跑一次；
    # 如果用来进行train encoder的话，每个样本单独forward会不会有影响呢？
    # 没有影响的，因为encoder的数据是每个batch一起来的；
    loss_batch = []
    correct_list_batch = []
    criterion = torch.nn.CrossEntropyLoss()
    for idx in range(len(input_batch)):

  
        for idx_np in range(len(num_pos[idx])):
            num_p = num_pos[idx][idx_np]
            # print(nums_batch[idx])

            if '%' in nums_batch[idx][idx_np]:
                target = torch.tensor(0, device='cuda').unsqueeze(dim=0)
            elif '/' in nums_batch[idx][idx_np]:
                target = torch.tensor(1, device='cuda').unsqueeze(dim=0)
            elif '.' in nums_batch[idx][idx_np]:
                target = torch.tensor(2, device='cuda').unsqueeze(dim=0)
            else:
                target = torch.tensor(3, device='cuda').unsqueeze(dim=0)
            # if target.item() > 10. or target.item() < -10:
            #     continue
            input_x = encoder_outputs[num_p][idx].unsqueeze(dim=0)

            pred = probing_type_module(input_x)
            # pdb.set_trace()
            loss_np = criterion(pred,target)
            correct_list_batch.append((torch.max(pred,1)[1]==target).item())
            loss_batch.append(loss_np)

        
    loss = sum(loss_batch) / len(loss_batch)

    probing_type_optim.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(probing_regression_module.parameters(),clip_value=1.1)
    probing_type_optim.step()
    return loss.item(),correct_list_batch

def test_probing_type(input_batch, input_length,output_batch, output_length, encoder, probing_type_module, probing_type_optim,
               nums_batch, num_pos,output_lang):

    # print(nums_batch)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    encoder.eval()
    probing_type_module.eval()
    if USE_CUDA:
        input_var = input_var.cuda()
    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs =  encoder_outputs.detach()


    # 进行数字的类别分类；float；int;分数；百分数；

    # 需要每一个样本跑一次；
    # 如果用来进行train encoder的话，每个样本单独forward会不会有影响呢？
    # 没有影响的，因为encoder的数据是每个batch一起来的；
    loss_batch = []
    correct_list_batch = []
    criterion = torch.nn.CrossEntropyLoss()
    correct_list_opter = [[],[],[],[]]   # 统计每个类别的正确率
    for idx in range(len(input_batch)):

  
        for idx_np in range(len(num_pos[idx])):
            num_p = num_pos[idx][idx_np]
            # print(nums_batch[idx])

            if '%' in nums_batch[idx][idx_np]:
                target = torch.tensor(0, device='cuda').unsqueeze(dim=0)
            elif '/' in nums_batch[idx][idx_np]:
                target = torch.tensor(1, device='cuda').unsqueeze(dim=0)
            elif '.' in nums_batch[idx][idx_np]:
                target = torch.tensor(2, device='cuda').unsqueeze(dim=0)
            else:
                target = torch.tensor(3, device='cuda').unsqueeze(dim=0)
            # if target.item() > 10. or target.item() < -10:
            #     continue
            input_x = encoder_outputs[num_p][idx].unsqueeze(dim=0)

            pred = probing_type_module(input_x)
            correct_list_batch.append((torch.max(pred,1)[1]==target).item())
            correct_list_opter[target.item()].append((torch.max(pred,1)[1]==target).item())
            
            # pdb.set_trace()
            loss_np = criterion(pred,target)
            loss_batch.append(loss_np)

        
    loss = sum(loss_batch) / len(loss_batch)

    return loss.item(),correct_list_batch, correct_list_opter

def train_probing_regression(input_batch, input_length,output_batch, output_length, encoder, probing_regression_module, probing_regression_optim,
               nums_batch, num_pos,output_lang):

    # print(nums_batch)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    encoder.eval()
    probing_regression_module.train()
    if USE_CUDA:
        input_var = input_var.cuda()
    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs =  encoder_outputs.detach()

    # 需要每一个样本跑一次；
    # 如果用来进行train encoder的话，每个样本单独forward会不会有影响呢？
    # 没有影响的，因为encoder的数据是每个batch一起来的；
    loss_batch = []
    criterion = torch.nn.MSELoss()
    for idx in range(len(input_batch)):
        
        # if idx == 1:
        #     print(input_batch[idx])
        #     print(nums_batch[idx])
        #     print(num_pos[idx])
        # print(output_batch[idx][0:output_length[idx]])

        FLAG = False
        for num in nums_batch[idx]:
            if NUM_to_float(num) > 100.0 or  NUM_to_float(num)<-100.0:
                FLAG = True
        if FLAG:
            continue

  
        for idx_np in range(len(num_pos[idx])):
            num_p = num_pos[idx][idx_np]
            # print(nums_batch[idx])
            target = torch.tensor(NUM_to_float(nums_batch[idx][idx_np]),device='cuda').unsqueeze(dim=0).unsqueeze(dim=0)
            # if target.item() > 10. or target.item() < -10:
            #     continue
            input_x = encoder_outputs[num_p][idx].unsqueeze(dim=0)

            pred = probing_regression_module(input_x)
            # pdb.set_trace()
            loss_np = criterion(pred,target)
            loss_batch.append(loss_np)

        
    loss = sum(loss_batch) / len(loss_batch)

    probing_regression_optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(probing_regression_module.parameters(),clip_value=1.1)
    probing_regression_optim.step()
    return loss.item()

def test_probing_regression(input_batch, input_length,output_batch, output_length, encoder, probing_regression_module, probing_regression_optim,
               nums_batch, num_pos,output_lang):

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    encoder.eval()
    probing_regression_module.eval()
    if USE_CUDA:
        input_var = input_var.cuda()
    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs = encoder_outputs.detach()

    # 需要每一个样本跑一次；
    # 如果用来进行train encoder的话，每个样本单独forward会不会有影响呢？
    # 没有影响的，因为encoder的数据是每个batch一起来的；
    loss_batch = []
    criterion = torch.nn.MSELoss()
    for idx in range(len(input_batch)):
        
        # print(output_batch[idx])
        # print(output_length[idx])

        FLAG = False
        for num in nums_batch[idx]:
            if NUM_to_float(num) > 100.0 or  NUM_to_float(num)<-100.0:
                FLAG = True
        if FLAG:
            continue
  
        for idx_np in range(len(num_pos[idx])):
            num_p = num_pos[idx][idx_np]
            target = torch.tensor(NUM_to_float(nums_batch[idx][idx_np]),device='cuda').unsqueeze(dim=0)
            input_x = encoder_outputs[num_p][idx].unsqueeze(dim=0)

            # if target.item() > 10. or target.item() < -10:
            #     continue

            pred = probing_regression_module(input_x)
            # print('pred: '+str(pred))
            # print('target: '+str(target))
            loss_np = criterion(pred,target)
            loss_batch.append(loss_np)

        
    loss = sum(loss_batch) / len(loss_batch)

    return loss.item()


def test_probing_regression_random(input_batch, input_length,output_batch, output_length, encoder, probing_regression_module, probing_regression_optim,
               nums_batch, num_pos,output_lang):

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    encoder.eval()
    probing_regression_module.eval()
    if USE_CUDA:
        input_var = input_var.cuda()
    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs = encoder_outputs.detach()

    # 需要每一个样本跑一次；
    # 如果用来进行train encoder的话，每个样本单独forward会不会有影响呢？
    # 没有影响的，因为encoder的数据是每个batch一起来的；
    loss_batch = []
    criterion = torch.nn.MSELoss()
    for idx in range(len(input_batch)):
        
        # print(output_batch[idx])
        # print(output_length[idx])

        FLAG = False
        for num in nums_batch[idx]:
            if NUM_to_float(num) > 100.0 or  NUM_to_float(num)<-100.0:
                FLAG = True
        if FLAG:
            continue
  
        num_pos_copy = num_pos[idx].copy()
        random.shuffle(num_pos_copy)
        for idx_np in range(len(num_pos_copy)):
            
            num_p = num_pos_copy[idx_np]
            target = torch.tensor(NUM_to_float(nums_batch[idx][idx_np]),device='cuda').unsqueeze(dim=0)
            input_x = encoder_outputs[num_p][idx].unsqueeze(dim=0)

            # if target.item() > 10. or target.item() < -10:
            #     continue

            pred = probing_regression_module(input_x)
            loss_np = criterion(pred,target)
            loss_batch.append(loss_np)

        
    loss = sum(loss_batch) / len(loss_batch)

    return loss.item()

def train_probing_opter(input_batch, input_length,output_batch, output_length, encoder, probing_opter_module, probing_opter_optim,
               nums_batch, num_pos,output_lang):

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    # TODO
    encoder.train()
    probing_opter_module.train()
    if USE_CUDA:
        input_var = input_var.cuda()
    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    pdb.set_trace()
    encoder_outputs = encoder_outputs.detach()

    # 需要每一个样本跑一次；
    # 如果用来进行train encoder的话，每个样本单独forward会不会有影响呢？
    # 没有影响的，因为encoder的数据是每个batch一起来的；
    loss_batch = []
    correct_list_batch = []
    for idx in range(len(input_batch)):
        
        # print(output_batch[idx])
        # print(output_length[idx])
        # 如果公式长度小于2，则过滤此样本；
        if output_length[idx] < 2:
            continue
        
        stack, equation,Num_list = compute_tree_distance(output_batch[idx][0:output_length[idx]],output_lang)
        # # pdb.set_trace()
        # if len(Num_list) < 2:
        #     continue
        if 'UNK' in equation:
            continue
        # try:
        #     dist_dict = stack.base[0].dist
        # except:
        #     continue

        criterion = torch.nn.CrossEntropyLoss()

        loss_plm = []
        for idx_e in range(len(equation)-2):
            c = equation[idx_e]
            if c in output_lang.index2word[0:4]:
                if equation[idx_e+1] in output_lang.index2word[7:22]:
                    if equation[idx_e+2] in output_lang.index2word[7:22]:
                        num_i_pos = num_pos[idx][int(equation[idx_e+1].replace('N',''))]
                        num_j_pos = num_pos[idx][int(equation[idx_e+2].replace('N',''))]

                        feature_i = encoder_outputs[num_i_pos][idx].unsqueeze(dim=0)
                        feature_j = encoder_outputs[num_j_pos][idx].unsqueeze(dim=0)
                        # pdb.set_trace()
                        logits = probing_opter_module(feature_i,feature_j)
                        # pdb.set_trace()
                        _,predict=torch.max(logits,dim=1)
                        # pdb.set_trace()
                        correct_list_batch.append(predict==output_lang.word2index[c])
                        loss_plm.append(criterion(logits,torch.tensor(output_lang.word2index[c]).view(1).cuda()))
        # loss_batch.append(sum(loss_plm) / len(loss_plm))
        # pdb.set_trace()
        if len(loss_plm) == 0:
            continue
        loss_batch.append(sum(loss_plm) / len(loss_plm))
    loss = sum(loss_batch) / len(loss_batch)

    probing_opter_optim.zero_grad()
    loss.backward()
    probing_opter_optim.step()
    return loss.item(),correct_list_batch

def test_probing_opter(input_batch, input_length,output_batch, output_length, encoder, probing_opter_module, probing_opter_optim,
               nums_batch, num_pos,output_lang):

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    encoder.eval()
    probing_opter_module.eval()
    if USE_CUDA:
        input_var = input_var.cuda()
    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs = encoder_outputs.detach()

    # 需要每一个样本跑一次；
    # 如果用来进行train encoder的话，每个样本单独forward会不会有影响呢？
    # 没有影响的，因为encoder的数据是每个batch一起来的；
    loss_batch = []
    correct_list_batch = []
    for idx in range(len(input_batch)):
        
        # print(output_batch[idx])
        # print(output_length[idx])
        # 如果公式长度小于2，则过滤此样本；
        if output_length[idx] < 2:
            continue
        
        stack, equation,Num_list = compute_tree_distance(output_batch[idx][0:output_length[idx]],output_lang)
        # # pdb.set_trace()
        # if len(Num_list) < 2:
        #     continue
        if 'UNK' in equation:
            continue
        # try:
        #     dist_dict = stack.base[0].dist
        # except:
        #     continue

        criterion = torch.nn.CrossEntropyLoss()

        loss_plm = []
        for idx_e in range(len(equation)-2):
            c = equation[idx_e]
            if c in output_lang.index2word[0:4]:
                if equation[idx_e+1] in output_lang.index2word[7:22]:
                    if equation[idx_e+2] in output_lang.index2word[7:22]:
                        num_i_pos = num_pos[idx][int(equation[idx_e+1].replace('N',''))]
                        num_j_pos = num_pos[idx][int(equation[idx_e+2].replace('N',''))]

                        feature_i = encoder_outputs[num_i_pos][idx].unsqueeze(dim=0)
                        feature_j = encoder_outputs[num_j_pos][idx].unsqueeze(dim=0)
                        # pdb.set_trace()
                        logits = probing_opter_module(feature_i,feature_j)
                        # pdb.set_trace()
                        _,predict=torch.max(logits,dim=1)
                        # pdb.set_trace()
                        correct_list_batch.append(predict==output_lang.word2index[c])
                        loss_plm.append(criterion(logits,torch.tensor(output_lang.word2index[c]).view(1).cuda()))
        # loss_batch.append(sum(loss_plm) / len(loss_plm))
        # pdb.set_trace()
        if len(loss_plm) == 0:
            continue
        loss_batch.append(sum(loss_plm) / len(loss_plm))
    loss = sum(loss_batch) / len(loss_batch)

    # probing_opter_optim.zero_grad()
    # loss.backward()
    # probing_opter_optim.step()
    return loss.item(),correct_list_batch

# '''
def train_probing_distance(input_batch, input_length,output_batch, output_length, encoder, probing_distance_module, probing_distance_optim,
               nums_batch, num_pos,output_lang):

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    # TODO
    encoder.train()
    probing_distance_module.train()
    if USE_CUDA:
        input_var = input_var.cuda()
    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs = encoder_outputs.detach()

    # 需要每一个样本跑一次；
    # 如果用来进行train encoder的话，每个样本单独forward会不会有影响呢？
    # 没有影响的，因为encoder的数据是每个batch一起来的；
    loss_batch = []
    for idx in range(len(input_batch)):
        
        # print(output_batch[idx])
        # print(output_length[idx])
        if output_length[idx] < 2:
            continue
        stack, equation,Num_list = compute_tree_distance(output_batch[idx][0:output_length[idx]],output_lang)
        # pdb.set_trace()
        if len(Num_list) < 2:
            continue
        if 'UNK' in equation:
            continue
        try:
            dist_dict = stack.base[0].dist
        except:
            continue

       
        # print(equation)
        # print(dist_dict)

        loss_pbl = []
        for i in range(len(Num_list)):
            for j in range(i+1, len(Num_list)):
                num_i_pos = num_pos[idx][int(Num_list[i].replace('N',''))]
                num_j_pos = num_pos[idx][int(Num_list[j].replace('N',''))]
                feature_i = encoder_outputs[num_i_pos][idx]
                feature_j = encoder_outputs[num_j_pos][idx]

                edges = []
                for k,v in dist_dict.items():
                    if k == Num_list[i] or k.find(Num_list[i]+'__')!= -1:
                        for kk, vv in v.items():
                            if kk == Num_list[j] or kk.find(Num_list[j]+'__') != -1:
                                edges.append(vv)
                distance_tree = sum(edges) / len(edges)
                # print(str(Num_list[i])+'   '+str(Num_list[j])+': '+str(distance_tree))

                dist_feautre = probing_distance_module(feature_i,feature_j)
                loss_pbl.append(torch.abs(dist_feautre - distance_tree))

        # pdb.set_trace()
        if len(loss_pbl) == 0:
            pdb.set_trace()
        loss_batch.append(sum(loss_pbl) / len(loss_pbl))
    loss = sum(loss_batch) / len(loss_batch)
    probing_distance_optim.zero_grad()
    loss.backward()
    probing_distance_optim.step()
    return loss.item()


def test_probing_distance(input_batch, input_length,output_batch, output_length, encoder, probing_distance_module, probing_distance_optim,
               nums_batch, num_pos,output_lang):

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    encoder.eval()
    probing_distance_module.eval()
    if USE_CUDA:
        input_var = input_var.cuda()
    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs = encoder_outputs.detach()

    # 需要每一个样本跑一次；
    # 如果用来进行train encoder的话，每个样本单独forward会不会有影响呢？
    # 没有影响的，因为encoder的数据是每个batch一起来的；
    loss_batch = []
    for idx in range(len(input_batch)):
        
        if output_length[idx] < 2:
            continue
        stack,equation,Num_list = compute_tree_distance(output_batch[idx][0:output_length[idx]],output_lang)
        # pdb.set_trace()
        if len(Num_list) < 2:
            continue
        if 'UNK' in equation:
            continue
        try:
            dist_dict = stack.base[0].dist
        except:
            continue

        loss_pbl = []
        for i in range(len(Num_list)):
            for j in range(i+1, len(Num_list)):
                num_i_pos = num_pos[idx][int(Num_list[i].replace('N',''))]
                num_j_pos = num_pos[idx][int(Num_list[j].replace('N',''))]
                feature_i = encoder_outputs[num_i_pos][idx]
                feature_j = encoder_outputs[num_j_pos][idx]

                edges = []
                for k,v in dist_dict.items():
                    if k == Num_list[i] or k.find(Num_list[i]+'__')!= -1:
                        for kk, vv in v.items():
                            if kk == Num_list[j] or kk.find(Num_list[j]+'__') != -1:
                                edges.append(vv)
                distance_tree = sum(edges) / len(edges)

                dist_feautre = probing_distance_module(feature_i,feature_j)
                loss_pbl.append(torch.abs(dist_feautre - distance_tree))

        loss_batch.append(sum(loss_pbl) / len(loss_pbl))
    loss = sum(loss_batch) / len(loss_batch)
    # probing_distance_optim.zero_grad()
    # loss.backward()
    # probing_distance_optim.step()
    return loss.item()

def test_probing_distance_random(input_batch, input_length,output_batch, output_length, encoder, probing_distance_module, probing_distance_optim,
               nums_batch, num_pos,output_lang):

    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    encoder.eval()
    probing_distance_module.eval()
    if USE_CUDA:
        input_var = input_var.cuda()
    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs = encoder_outputs.detach()

    # 需要每一个样本跑一次；
    # 如果用来进行train encoder的话，每个样本单独forward会不会有影响呢？
    # 没有影响的，因为encoder的数据是每个batch一起来的；
    loss_batch = []
    for idx in range(len(input_batch)):
        
        if output_length[idx] < 2:
            continue
        stack,equation,Num_list = compute_tree_distance(output_batch[idx][0:output_length[idx]],output_lang)
        # pdb.set_trace()
        if len(Num_list) < 2:
            continue
        if 'UNK' in equation:
            continue
        try:
            dist_dict = stack.base[0].dist
        except:
            continue

        loss_pbl = []
        random_list = [i for i in range(len(num_pos[idx]))]
        for i in range(len(Num_list)):
            for j in range(i+1, len(Num_list)):
                # num_i_pos = num_pos[idx][int(Num_list[i].replace('N',''))]
                # num_j_pos = num_pos[idx][int(Num_list[j].replace('N',''))]
                num_ij_pos = random.sample(random_list,2)
                feature_i = encoder_outputs[num_ij_pos[0]][idx]
                feature_j = encoder_outputs[num_ij_pos[1]][idx]

                edges = []
                for k,v in dist_dict.items():
                    if k == Num_list[i] or k.find(Num_list[i]+'__')!= -1:
                        for kk, vv in v.items():
                            if kk == Num_list[j] or kk.find(Num_list[j]+'__') != -1:
                                edges.append(vv)
                distance_tree = sum(edges) / len(edges)

                dist_feautre = probing_distance_module(feature_i,feature_j)
                loss_pbl.append(torch.abs(dist_feautre - distance_tree))

        loss_batch.append(sum(loss_pbl) / len(loss_pbl))
    loss = sum(loss_batch) / len(loss_batch)
    # probing_distance_optim.zero_grad()
    # loss.backward()
    # probing_distance_optim.step()
    return loss.item()


# '''

def train_probing_compare(input_batch, input_length, encoder, probing_compare_module, probing_compare_optim,
               nums_batch, num_pos):


    # 处理数据，组成每个batch的适合probing的数据：从num_pos的长度中抽取2个位置，然后抽取出对应的num和pos；
    # 这个num代表着比较的target，pos代表着可以从contextual representation中抽取num的表征；

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    # target = torch.LongTensor(target_batch).transpose(0, 1)

    # padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    # batch_size = len(input_length)

    # TODO
    encoder.train()
    probing_compare_module.train()
    # predict.train()
    # generate.train()
    # merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()


    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs = encoder_outputs.detach()

    cpair_pos_batch = []
    cpair_num_batch = []                  # 这是一个batch的数据，每一个是一个数学题中，抽出的那两个数字
    cpair_input_feature_batch_left = []        # 这是一个batch的数据，每一个是一个数学题中，抽出的那两个数字的特征
    cpair_input_feature_batch_right = [] 
    for i in range(len(input_batch)):
        if len(num_pos[i]) == 1:
            pair_pos = [0,0]
        else:
            pair_pos = random.sample(range(0,len(num_pos[i])),2)

        cpair_pos_batch.append(pair_pos)

        cpair_num_batch_temp = []
        for j in range(2):
            cpair_num_batch_temp.append(nums_batch[i][pair_pos[j]])
            if j == 0:
                cpair_input_feature_batch_left.append(encoder_outputs[num_pos[i][pair_pos[j]],i,:])
            else:
                cpair_input_feature_batch_right.append(encoder_outputs[num_pos[i][pair_pos[j]],i,:])
        cpair_num_batch.append(cpair_num_batch_temp)


    # print(cpair_num_batch)
    for i in range(len(cpair_num_batch)):
        for j in range(2):
            cpair_num_batch[i][j] = cpair_num_batch[i][j].replace(')','').replace('(','')
            if cpair_num_batch[i][j].find('/') != -1:
                cpair_num_batch[i][j] = float(Fraction(cpair_num_batch[i][j]))
            elif cpair_num_batch[i][j].find('%') != -1:
                cpair_num_batch[i][j] = cpair_num_batch[i][j].replace('%','')
                cpair_num_batch[i][j] = float(cpair_num_batch[i][j])/100.
            else:
                cpair_num_batch[i][j] = float(cpair_num_batch[i][j])
    
    probing_comp_target_batch = []
    for i in range(len(cpair_num_batch)):
        probing_comp_target_batch.append(1. if cpair_num_batch[i][0] > cpair_num_batch[i][1] else 0.)
    probing_comp_target_batch_tensor = torch.tensor(probing_comp_target_batch).unsqueeze(dim=1).cuda()


    left_contextual_vector = torch.stack(cpair_input_feature_batch_left)
    right_contextual_vector = torch.stack(cpair_input_feature_batch_right)

    outputs=probing_compare_module(left_contextual_vector,right_contextual_vector)

    # pdb.set_trace()
    correct_sum = ((torch.nn.functional.sigmoid(outputs) > 0.5) == probing_comp_target_batch_tensor).sum()

    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(outputs,probing_comp_target_batch_tensor)
    # pdb.set_trace()
    probing_compare_optim.zero_grad()
    loss.backward()
    probing_compare_optim.step()
    # 现在，我们需要从每个句子中，抽出两个num，然后根据对应vector(encoder_outputs中来的)，去预测二者大小关系；
    # 同时，我们也需要计算出，二者本来的关系；这样的话，我们可能需要在前面处理数据；然后按照batch传入？
    return loss.item(), correct_sum  # , loss_0.item(), loss_1.item()


def test_probing_compare(input_batch, input_length, encoder, probing_compare_module, probing_compare_optim,
               nums_batch, num_pos):


    # 处理数据，组成每个batch的适合probing的数据：从num_pos的长度中抽取2个位置，然后抽取出对应的num和pos；
    # 这个num代表着比较的target，pos代表着可以从contextual representation中抽取num的表征；

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    # target = torch.LongTensor(target_batch).transpose(0, 1)

    # padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    # batch_size = len(input_length)

    encoder.eval()
    probing_compare_module.eval()
    # predict.train()
    # generate.train()
    # merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()


    encoder_outputs, _ = encoder(input_var, input_length) # encoder_outputa S x B x H
    # pdb.set_trace()
    encoder_outputs = encoder_outputs.detach()

    cpair_pos_batch = []
    cpair_num_batch = []                  # 这是一个batch的数据，每一个是一个数学题中，抽出的那两个数字
    cpair_input_feature_batch_left = []        # 这是一个batch的数据，每一个是一个数学题中，抽出的那两个数字的特征
    cpair_input_feature_batch_right = [] 
    for i in range(len(input_batch)):
        if len(num_pos[i]) == 1:
            pair_pos = [0,0]
        else:
            pair_pos = random.sample(range(0,len(num_pos[i])),2)

        cpair_pos_batch.append(pair_pos)

        cpair_num_batch_temp = []
        for j in range(2):
            cpair_num_batch_temp.append(nums_batch[i][pair_pos[j]])
            if j == 0:
                cpair_input_feature_batch_left.append(encoder_outputs[num_pos[i][pair_pos[j]],i,:])
            else:
                cpair_input_feature_batch_right.append(encoder_outputs[num_pos[i][pair_pos[j]],i,:])
        cpair_num_batch.append(cpair_num_batch_temp)


    # print(cpair_num_batch)
    for i in range(len(cpair_num_batch)):
        for j in range(2):
            cpair_num_batch[i][j] = cpair_num_batch[i][j].replace(')','').replace('(','')
            if cpair_num_batch[i][j].find('/') != -1:
                cpair_num_batch[i][j] = float(Fraction(cpair_num_batch[i][j]))
            elif cpair_num_batch[i][j].find('%') != -1:
                cpair_num_batch[i][j] = cpair_num_batch[i][j].replace('%','')
                cpair_num_batch[i][j] = float(cpair_num_batch[i][j])/100.
            else:
                cpair_num_batch[i][j] = float(cpair_num_batch[i][j])
    
    probing_comp_target_batch = []
    for i in range(len(cpair_num_batch)):
        probing_comp_target_batch.append(1. if cpair_num_batch[i][0] > cpair_num_batch[i][1] else 0.)
    probing_comp_target_batch_tensor = torch.tensor(probing_comp_target_batch).unsqueeze(dim=1).cuda()


    left_contextual_vector = torch.stack(cpair_input_feature_batch_left)
    right_contextual_vector = torch.stack(cpair_input_feature_batch_right)

    outputs=probing_compare_module(left_contextual_vector,right_contextual_vector)

    # pdb.set_trace()
    correct_sum = ((torch.nn.functional.sigmoid(outputs) > 0.5) == probing_comp_target_batch_tensor).sum()

    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(outputs,probing_comp_target_batch_tensor)
    # pdb.set_trace()
    # probing_compare_optim.zero_grad()
    # loss.backward()
    # probing_compare_optim.step()
    # 现在，我们需要从每个句子中，抽出两个num，然后根据对应vector(encoder_outputs中来的)，去预测二者大小关系；
    # 同时，我们也需要计算出，二者本来的关系；这样的话，我们可能需要在前面处理数据；然后按照batch传入？
    return loss.item(), correct_sum  # , loss_0.item(), loss_1.item()


def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos,
                  beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out


def topdown_train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch,
                       generate_nums, encoder, predict, generate, encoder_optimizer, predict_optimizer,
                       generate_optimizer, output_lang, num_pos, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        for idx, l, r, node_stack, i in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                            node_stacks, target[t].tolist()):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def topdown_evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, output_lang, num_pos,
                          beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, embeddings_stacks, left_childs,
                                              current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out
