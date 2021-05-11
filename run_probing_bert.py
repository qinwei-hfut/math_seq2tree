# coding: utf-8
import warnings
warnings.filterwarnings('ignore')
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
import pdb

batch_size = 64
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2

data = load_raw_data("data/Math_23K.json")

pairs, generate_nums, copy_nums = transfer_num(data)


# number_n_gram_word(pairs,n_gram=3)
temp_pairs = []
for p in pairs:
    # 将equation的表达换成了前缀表达；
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
# pdb.set_trace()
pairs = temp_pairs



fold_size = int(len(pairs) * 0.2)
fold_pairs = []
for split_fold in range(4):
    fold_start = fold_size * split_fold
    fold_end = fold_size * (split_fold + 1)
    fold_pairs.append(pairs[fold_start:fold_end])
fold_pairs.append(pairs[(fold_size * 4):])

best_acc_fold = []

# torch.cuda.set_device(0)

for fold in range(5):
    if fold < 4:
        continue
    pairs_tested = []
    pairs_trained = []
    for fold_t in range(5):
        if fold_t == fold:
            pairs_tested += fold_pairs[fold_t]
        else:
            pairs_trained += fold_pairs[fold_t]

    # pairs_trained he pairs_tested 就是通过fold选择后的pairs
    # train_pairs 将题干的文字和后缀表达式数字化了；增加了input_len和output_len

    input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                    copy_nums, tree=True)
    
    pairs_trained, pairs_tested = prepare_data_for_bert(pairs_trained, pairs_tested)
    # pdb.set_trace()
    # Initialize models
    encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                         n_layers=n_layers)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                         input_size=len(generate_nums))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                            embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)

    # ----- -----TODO hidden size
    probing_compare_module = Probing_Compare_Module(embedding_size=768,hidden_size= 200,linear=False,cat=True)
    probing_distance_module = Probing_Distance_Module(embedding_size=768,hidden_size=256)
    probing_opter_module = Probing_Opter_Module(embedding_size=768,hidden_size= 200,linear=False,cat=True)
    probing_regression_module = Probing_Regression_Module(embedding_size=768,hidden_dim=200,linear=False)
    probing_type_module = Probing_Type_Module(embedding_size=768,hidden_dim=200,linear=True)
    # the embedding layer is  only for generated number embeddings, operators, and paddings

    # encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
    probing_compare_optim = torch.optim.SGD(probing_compare_module.parameters(), lr=0.01, momentum=0.5)
    probing_distance_optim = torch.optim.SGD(probing_distance_module.parameters(), lr=0.001, momentum=0.5)
    probing_opter_optim = torch.optim.SGD(probing_opter_module.parameters(), lr=0.01, momentum=0.5)
    probing_regression_optim = torch.optim.SGD(probing_regression_module.parameters(), lr=0.01, momentum=0.5)
    probing_type_optim = torch.optim.SGD(probing_type_module.parameters(), lr=0.01, momentum=0.5)

    # encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
    # predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
    # generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
    # merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)
    # TODO super parameter ?
    probing_compare_scheduler = torch.optim.lr_scheduler.StepLR(probing_compare_optim, step_size=20,gamma=0.5)
    probing_distance_scheduler = torch.optim.lr_scheduler.StepLR(probing_distance_optim, step_size=20,gamma=0.5)
    probing_opter_scheduler = torch.optim.lr_scheduler.StepLR(probing_opter_optim, step_size=20,gamma=0.5)
    probing_regression_scheduler = torch.optim.lr_scheduler.StepLR(probing_regression_optim,step_size=20,gamma=0.5)
    probing_type_scheduler = torch.optim.lr_scheduler.StepLR(probing_type_optim,step_size=20,gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()
        probing_compare_module.cuda()
        probing_distance_module.cuda()
        probing_opter_module.cuda()
        probing_regression_module.cuda()
        probing_type_module.cuda()

    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])


    encoder.load_state_dict(torch.load('./models/encoder'))
    predict.load_state_dict(torch.load('./models/predict'))
    generate.load_state_dict(torch.load('./models/generate'))
    merge.load_state_dict(torch.load('./models/merge'))


    test_loaded_model = False
    if test_loaded_model:
        print('test mwp model:')
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                        merge, output_lang, test_batch[5], beam_size=beam_size)
            # pdb.set_trace()
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
    


    print('train probing compare task')
    best_test_acc = 0
    for epoch in range(n_epochs):
        # encoder_scheduler.step()
        # predict_scheduler.step()
        # generate_scheduler.step()
        # merge_scheduler.step()
        
        # input_batches 的第一个dim是选择哪一个batch，bs=64的情况下，batch数量是290;
        # 第二个dim就是在batch中选择样本；第三个dim就是每个样本的vector
        # input_lengths； 在上面的每个batch的数据中，每个样本的vector长度都保持了一致，通过补0和该batch的最长的vector的长度一致；
        # 因此，input_lengths就是标记了每个样本的实际长度；二维的；
        # input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_pos_batches, num_size_batches =prepare_train_batch_for_bert(pairs_trained,batch_size)
        # pdb.set_trace()
        print("fold:", fold + 1)
        print("epoch:", epoch + 1)

        '''
        # ######### evaluate probing distance task
        print('evaluate probing distance task:')
        loss_total_test = 0
        loss_total_test_random=0
        # input_batches 的第一个dim是选择哪一个batch，bs=64的情况下，batch数量是290;
        # 第二个dim就是在batch中选择样本；第三个dim就是每个样本的vector
        # input_lengths； 在上面的每个batch的数据中，每个样本的vector长度都保持了一致，通过补0和该batch的最长的vector的长度一致；
        # 因此，input_lengths就是标记了每个样本的实际长度；二维的；
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(test_pairs, batch_size)
        # pdb.set_trace()
        start = time.time()
        
        for idx in range(len(input_lengths)):
            loss_dist = test_probing_distance(input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx], encoder, probing_distance_module, probing_distance_optim, nums_batches[idx], num_pos_batches[idx],output_lang)
            loss_total_test += loss_dist

            loss_dist_random = test_probing_distance_random(input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx], encoder, probing_distance_module, probing_distance_optim, nums_batches[idx], num_pos_batches[idx],output_lang)
            loss_total_test_random += loss_dist_random
            # print('test loss batch '+str(idx)+': '+str(loss_dist))
        print("test loss:", loss_total_test / len(input_lengths))
        print("test loss random:", loss_total_test_random / len(input_lengths))
        print("test time", time_since(time.time() - start))
        print("--------------------------------")
        ##
        #################
        '''


        ###########################
        print("training probing task:")
        start = time.time()
        loss_total = 0
        correct_list_total = []
        correct_total = 0
        # print(len(input_lengths))
        # print(len(input_batches))
        for idx in range(len(input_lengths)):
            if idx % 50 == 0:
                print('train '+str(idx))
            # if idx == 20:
            #     break

            '''
            loss_probing_compare, correct_sum = train_probing_compare(input_batches[idx], input_lengths[idx], encoder, probing_compare_module, probing_compare_optim, nums_batches[idx], num_pos_batches[idx])
            loss_total += loss_probing_compare
            correct_total += correct_sum
            '''

            
            loss_dist = train_probing_distance_bert(input_batches[idx], input_lengths[idx], output_batches[idx],output_lengths[idx], probing_opter_module, probing_opter_optim, nums_batches[idx], num_pos_batches[idx],output_lang)
            loss_total += loss_dist
            

            '''
            # opter
            loss_probing_compare, correct_list_batch = train_probing_opter_bert(input_batches[idx], input_lengths[idx], output_batches[idx],output_lengths[idx], probing_opter_module, probing_opter_optim, nums_batches[idx], num_pos_batches[idx],output_lang)
            loss_total += loss_probing_compare
            correct_list_total += correct_list_batch
            '''
            


            # loss_probing_compare = train_probing_regression(input_batches[idx], input_lengths[idx], output_batches[idx],output_lengths[idx], encoder, probing_regression_module, probing_regression_optim, nums_batches[idx], num_pos_batches[idx],output_lang)
            # # print(loss_probing_compare)
            # loss_total += loss_probing_compare
            
            '''
            loss_probing_compare,correct_list_batch = train_probing_type_bert(input_batches[idx], input_lengths[idx], output_batches[idx],output_lengths[idx], probing_type_module, probing_type_optim, nums_batches[idx], num_pos_batches[idx],output_lang)
            # print(loss_probing_compare)
            loss_total += loss_probing_compare
            correct_list_total += correct_list_batch
            '''
        

        # pdb.set_trace()
        print("training loss:", loss_total / len(input_lengths))
        # print("training acc:", sum(correct_list_total) / float(len(correct_list_total)))
        print("training time", time_since(time.time() - start))
        print("--------------------------------")
        
        

        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_pos_batches, num_size_batches =prepare_train_batch_for_bert(pairs_tested,batch_size)
        '''
        ##### evaluate probing_compare()
        correct_total_test = 0
        for idx in range(len(input_lengths)):
            loss_probing_compare, correct_sum_test = test_probing_compare(input_batches[idx], input_lengths[idx], encoder, probing_compare_module, probing_compare_optim, nums_batches[idx], num_pos_batches[idx])
            loss_total += loss_probing_compare
            correct_total_test += correct_sum_test
        
        
        if  float(correct_total_test)/len(test_pairs) > best_test_acc:
            best_test_acc =  float(correct_total_test)/len(test_pairs)
        print("test loss:", loss_total / len(input_lengths))
        print("test acc:", float(correct_total_test)/len(test_pairs))
        print("training time", time_since(time.time() - start))
        print('best test acc:', best_test_acc)
        print("--------------------------------")
        '''

        
        '''
        ##### evaluate probing_opter()
        print('evaluate probing opter task:')
        start = time.time()
        correct_total_total_test = []
        for idx in range(len(input_lengths)):
            loss_probing_compare, correct_list_batch = test_probing_opter_bert(input_batches[idx], input_lengths[idx], output_batches[idx],output_lengths[idx], probing_opter_module, probing_opter_optim, nums_batches[idx], num_pos_batches[idx],output_lang)
            loss_total += loss_probing_compare
            correct_total_total_test += correct_list_batch
        
        
        # if  float(correct_total_test)/len(test_pairs) > best_test_acc:
        #     best_test_acc =  float(correct_total_test)/len(test_pairs)
        print("test loss:", loss_total / len(input_lengths))
        print("test acc:", sum(correct_total_total_test).item() / float(len(correct_total_total_test)))
        print("test time", time_since(time.time() - start))
        print("--------------------------------")
        '''
        


        '''
        ##### evaluate probing regression()
        print('evaluate probing regression task:')
        start = time.time()
        correct_total_total_test = []
        loss_total_random = []
        loss_total = []
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(test_pairs, batch_size)
        for idx in range(len(input_lengths)):
            loss_probing_compare = test_probing_regression(input_batches[idx], input_lengths[idx], output_batches[idx],output_lengths[idx], encoder, probing_regression_module, probing_regression_optim, nums_batches[idx], num_pos_batches[idx],output_lang)
            loss_total.append(loss_probing_compare)

            loss_probing_compare_random = test_probing_regression_random(input_batches[idx], input_lengths[idx], output_batches[idx],output_lengths[idx], encoder, probing_regression_module, probing_regression_optim, nums_batches[idx], num_pos_batches[idx],output_lang)
            loss_total_random.append(loss_probing_compare_random)
        
        
        # if  float(correct_total_test)/len(test_pairs) > best_test_acc:
        #     best_test_acc =  float(correct_total_test)/len(test_pairs)
        print("test loss:", sum(loss_total) / len(loss_total))
        print("test loss random:", sum(loss_total_random) / len(loss_total_random))
        print("test time", time_since(time.time() - start))
        print("--------------------------------")
        '''

        '''
        ##### evaluate probing type
        print('evaluate probing type:')
        start = time.time()
        correct_total_total_test = []
        correct_total_list_opter = [[],[],[],[]]
        loss_total_random = []
        loss_total = []
        # input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(test_pairs, batch_size)
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_pos_batches, num_size_batches =prepare_train_batch_for_bert(pairs_tested,batch_size)
        # print(len(input_lengths))
        for idx in range(len(input_lengths)):
            if idx == 50:
                print('test '+str(idx))

            loss_probing_type, correct_list_batch,correct_list_opter = test_probing_type_bert(input_batches[idx], input_lengths[idx], output_batches[idx],output_lengths[idx], probing_type_module, probing_type_optim, nums_batches[idx], num_pos_batches[idx],output_lang)
            loss_total.append(loss_probing_type)
            correct_total_total_test += correct_list_batch
            for i in range(len(correct_list_opter)):
                correct_total_list_opter[i] += correct_list_opter[i]


            
        
        
        # if  float(correct_total_test)/len(test_pairs) > best_test_acc:
        #     best_test_acc =  float(correct_total_test)/len(test_pairs)
        test_acc = ['percentage', 'fraction', 'float', 'int']
        for i in range(len(correct_total_list_opter)):
            print(test_acc[i]+': ', sum(correct_total_list_opter[i])/len(correct_total_list_opter[i]))
            print(len(correct_total_list_opter[i]))
        print("test loss:", sum(loss_total) / len(loss_total))
        print("test acc:", sum(correct_total_total_test) / len(correct_total_total_test))
        print("test time", time_since(time.time() - start))
        print("--------------------------------")
        '''
        
        


        


a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    print(best_acc_fold[bl])
print(a / float(c), b / float(c))
