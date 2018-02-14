
import torch
import transformer.Constants as Constants

import preprocess as prepro

# class Settings:

#     def __init__(self, train_src, train_tgt, valid_src, valid_tgt, save_data, max_word_seq_len, min_word_count, keep_case, share_vocab, vocab):
        
#         self.train_src = train_src
#         self.train_tgt = train_tgt
#         self.valid_src = valid_src
#         self.valid_tgt = valid_tgt
#         self.save_data = save_data
#         self.max_word_seq_len = max_word_seq_len
#         self.min_word_count = min_word_count
#         self.keep_case = keep_case
#         self.share_vocab = share_vocab
#         self.vocab = vocab

    

def prep(train_src, train_tgt, valid_src, valid_tgt, save_data, max_word_seq_len=50, min_word_count=5, keep_case=True, share_vocab=True, vocab=None):
    
    max_token_seq_len = max_word_seq_len + 2 # include the <s> and </s>

    # opt = Settings(train_src, train_tgt, valid_src, valid_tgt, save_data, max_word_seq_len, min_word_count, keep_case, share_vocab, vocab)

    # # Training set
    # train_src_word_insts = prepro.read_instances_from_file(
    #     train_src, max_word_seq_len, keep_case)
    # train_tgt_word_insts = prepro.read_instances_from_file(
    #     train_tgt, max_word_seq_len, keep_case)

    # if len(train_src_word_insts) != len(train_tgt_word_insts):
    #     print('[Warning] The training instance count is not equal.')
    #     min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
    #     train_src_word_insts = train_src_word_insts[:min_inst_count]
    #     train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    # #- Remove empty instances
    # train_src_word_insts, train_tgt_word_insts = list(zip(*[
    #     (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # # Validation set
    # valid_src_word_insts = prepro.read_instances_from_file(
    #     valid_src, max_word_seq_len, keep_case)
    # valid_tgt_word_insts = prepro.read_instances_from_file(
    #     valid_tgt, max_word_seq_len, keep_case)

    # if len(valid_src_word_insts) != len(valid_tgt_word_insts):
    #     print('[Warning] The validation instance count is not equal.')
    #     min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
    #     valid_src_word_insts = valid_src_word_insts[:min_inst_count]
    #     valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    # #- Remove empty instances
    # valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
    #     (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))


    src_word_insts = prepro.read_instances_from_file(
        train_src, max_word_seq_len, keep_case)
    tgt_word_insts = prepro.read_instances_from_file(
        train_tgt, max_word_seq_len, keep_case)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Build vocabulary
    if vocab:
        predefined_data = torch.load(vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = prepro.build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = prepro.build_vocab_idx(train_src_word_insts, min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = prepro.build_vocab_idx(train_tgt_word_insts, min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = prepro.convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = prepro.convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = prepro.convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = prepro.convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {
        'settings': max_token_seq_len,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', save_data)
    torch.save(data, save_data)
    print('[Info] Finish.')


