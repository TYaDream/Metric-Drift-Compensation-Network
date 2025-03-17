import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import tg1
import tg2
import scipy as sp
import scipy.stats
import random
import os
FEATURE_DIM = 40
LENGTH = 8
CLASS_NUM1 = 5
CLASS_NUM2 = 5
SAMPLE_NUM_PER_CLASS = 4
BATCH_NUM_PER_CLASS1 = 8
BATCH_NUM_PER_CLASS2 = 8
adapt_epochs = (SAMPLE_NUM_PER_CLASS - 1) * 24
EPISODE = 1401
TEST_EPISODE = 20
LEARNING_RATE = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1024 # any random number
accuracy_list = []
h_list = []
f1_list = []
train_mouths = [0]
test_mouths = [9]
# Transformer Parameters
d_model = 81  # Embedding Size
d_ff = 12 # FeedForward dimension
d_k = d_v = 6  # dimension of K(=Q), V
n_layers = 4  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def reset_Encoder_Batchnorm(net):
    net.BN0.reset_running_stats()
    net.BN1.reset_running_stats()
    net.BN2.reset_running_stats()
    net.BN3.reset_running_stats()
    net.BN0.reset_parameters()
    net.BN1.reset_parameters()
    net.BN2.reset_parameters()
    net.BN3.reset_parameters()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class se_block(nn.Module):
    def __init__(self, channel, ratio=4):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b,c)#[batch,8]
        y = self.fc(y).view(b,c,1)
        return x * y

class CNNEncoder(nn.Module):

    def __init__(self,init_weights = True):
        super(CNNEncoder, self).__init__()

        self.BN0 = nn.BatchNorm1d(16)
        self.BN1 = nn.BatchNorm1d(24)
        self.conv11 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv1d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv1d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv1d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.se1 = se_block(24)
        self.BN2 = nn.BatchNorm1d(32)
        self.conv21 = nn.Conv1d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv1d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv23 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv24 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv25 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.se2 = se_block(32)
        self.BN3 = nn.BatchNorm1d(40)
        self.conv31 = nn.Conv1d(in_channels=32, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv1d(in_channels=32, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.conv34 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.conv35 = nn.Conv1d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.se3 = se_block(40)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #print(x.shape)
        x = self.BN0(x)

        x1 = F.leaky_relu(self.BN1(self.conv11(x)))
        x1 = F.leaky_relu(self.BN1(self.conv13(x1)))
        x1 = F.leaky_relu(self.BN1(self.conv14(x1)))
        x1 = F.leaky_relu(self.BN1(self.conv15(x1)))
        x1 = self.se1(x1)
        x = x1 + self.BN1(self.conv12(x))

        x1 = F.leaky_relu(self.BN2(self.conv21(x)))
        x1 = F.leaky_relu(self.BN2(self.conv23(x1)))
        x1 = F.leaky_relu(self.BN2(self.conv24(x1)))
        x1 = F.leaky_relu(self.BN2(self.conv25(x1)))
        x1 = self.se2(x1)
        x = x1 + self.BN2(self.conv22(x))

        x1 = F.leaky_relu(self.BN3(self.conv31(x)))
        x1 = F.leaky_relu(self.BN3(self.conv33(x1)))
        x1 = F.leaky_relu(self.BN3(self.conv34(x1)))
        x1 = F.leaky_relu(self.BN3(self.conv35(x1)))
        x1 = self.se3(x1)
        x = x1 + self.BN3(self.conv32(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.bias, 0)
            # 是否为批归一化层
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class RelationNetwork(nn.Module):
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(in_features=8, out_features=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = torch.transpose(enc_inputs, 1, 2)
        #enc_outputs = enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        batch = enc_inputs.shape[0]
        x = torch.sigmoid(self.fc(enc_outputs[:,:,0]))

        return x

def swap_dim(samples,batches,other):
    batch_size,seq_len,dim_size = samples.size()

    dims_to_swap = torch.randperm(dim_size)[:2]

    samples_swapped = samples.clone()
    batches_swapped = batches.clone()
    other_swapped = other.clone()

    samples_swapped[:, :, dims_to_swap[0]], samples_swapped[:, :, dims_to_swap[1]] = \
        samples[:, :, dims_to_swap[1]], samples[:, :, dims_to_swap[0]]

    batches_swapped[:, :, dims_to_swap[0]], batches_swapped[:, :, dims_to_swap[1]] = \
        batches[:, :, dims_to_swap[1]], batches[:, :, dims_to_swap[0]]

    other_swapped[:, :, dims_to_swap[0]], other_swapped[:, :, dims_to_swap[1]] = \
        other[:, :, dims_to_swap[1]], other[:, :, dims_to_swap[0]]

    return samples_swapped,batches_swapped,other_swapped

def perturb_dim(samples,batches):
    batch_size1,seq_len,dim_size = samples.size()
    batch_size2,seq_len,dim_size = batches.size()

    perturb_factor = 1 + (torch.rand(1, seq_len, dim_size) - 0.5) * 0.4
    perturb_factor1 = perturb_factor.expand(batch_size1, -1, -1)
    perturb_factor2 = perturb_factor.expand(batch_size2, -1, -1)

    # 通过乘法应用扰动因子
    samples = samples * perturb_factor1
    batches = batches * perturb_factor2

    return samples,batches

def main():
    set_seed(SEED)
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders1,metatest_folders1 = tg1.mini_imagenet_folders()
    metatrain_folders2,metatest_folders2 = tg2.mini_imagenet_folders()
    # Step 2: init neural networks
    print("init neural networks")


    feature_encoder = CNNEncoder().to(device)
    relation_network = RelationNetwork().to(device)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    # feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=10,gamma=0.98)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    # relation_network_scheduler = StepLR(relation_network_optim,step_size=10,gamma=0.98)

    # 测试用
    feature_encoder_test = CNNEncoder().to(device)
    relation_network_test = RelationNetwork().to(device)

    feature_encoder_test_optim = torch.optim.Adam(feature_encoder_test.parameters(), lr=8 * LEARNING_RATE)
    relation_network_test_optim = torch.optim.Adam(relation_network_test.parameters(), lr=0 * LEARNING_RATE)
    mse = nn.MSELoss().to(device)
    # Step 3: build graph
    print("Training...")
    counter = 0#
    accumulator = 0.0#
    last_accuracy = 0.0
    for episode in range(EPISODE):

        # init dataset
        task1 = tg1.MiniImagenetTask(metatrain_folders1,CLASS_NUM1,train_mouths,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS1)
        sample_dataloader = tg1.get_mini_imagenet_data_loader(task1,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg1.get_mini_imagenet_data_loader(task1,num_per_class=BATCH_NUM_PER_CLASS1,split="test",shuffle=True)

        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().__next__()#[SAMPLE_NUM, 240, 8]
        batches,batch_labels = batch_dataloader.__iter__().__next__()#[BATCH_NUM, 240, 8]
        samples,batches,_ = swap_dim(samples,batches,batches)

        # calculate features
        sample_features = feature_encoder(Variable(samples.permute(0, 2, 1).to(torch.float32).to(device)))#[SAMPLE_NUM,64,30]
        batch_features = feature_encoder(Variable(batches.permute(0, 2, 1).to(torch.float32).to(device)))#[BATCH_NUM,64,30]
        sample_features = sample_features.view(CLASS_NUM1,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,LENGTH)
        sample_features = torch.mean(sample_features,1).squeeze(1)#[CLASS_NUM,64,30]
        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS1 * CLASS_NUM1, 1, 1, 1)
        # [BATCH_NUM,CLASS_NUM,64,30]
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM1, 1, 1, 1)
        # [CLASS_NUM,BATCH_NUM,64,30]
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        # [BATCH_NUM,CLASS_NUM,64,30]
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2, LENGTH)

        temp_shape = relation_pairs.shape;
        temp_tensor = torch.zeros([temp_shape[0], 1, 8]).to(device)
        relation_pairs = torch.cat((temp_tensor, relation_pairs), 1)

        # [BATCH_NUM*CLASS_NUM,FEATURE_DIM * 2,30]
        relations = relation_network(relation_pairs).view(-1, CLASS_NUM1)
        # [BATCH_NUM*CLASS_NUM,1]

        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS1*CLASS_NUM1, CLASS_NUM1).scatter_(1, batch_labels.view(-1,1), 1).to(device))

        loss = mse(relations,one_hot_labels)
        accumulator += loss.item()

        feature_encoder.zero_grad()
        relation_network.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode + 1) % 10 == 0:
            print("episode:", episode + 1, "loss", loss.data.data)

        if episode % 20 == 0 and episode >= 1400:
            torch.save(feature_encoder.state_dict(), 'feature_encoder.txt')
            torch.save(relation_network.state_dict(), 'relation_network.txt')
            f1_all = []
            # test
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):

                total_tp0 = 0;
                total_tp1 = 0;
                total_tp2 = 0;
                total_tp3 = 0;
                total_fn0 = 0;
                total_fn1 = 0;
                total_fn2 = 0;
                total_fn3 = 0;
                total_fp0 = 0;
                total_fp1 = 0;
                total_fp2 = 0;
                total_fp3 = 0;
                total_rewards = 0
                task2 = tg2.MiniImagenetTask(metatrain_folders2, CLASS_NUM2, test_mouths, SAMPLE_NUM_PER_CLASS,
                                             BATCH_NUM_PER_CLASS2)
                task1_sample_dataloader = tg1.get_mini_imagenet_data_loader(task1, num_per_class=SAMPLE_NUM_PER_CLASS,
                                                                            split="train", shuffle=False)
                sample_dataloader = tg2.get_mini_imagenet_data_loader(task2, num_per_class=SAMPLE_NUM_PER_CLASS,
                                                                      split="train", shuffle=False)
                test_dataloader = tg2.get_mini_imagenet_data_loader(task2, num_per_class=BATCH_NUM_PER_CLASS2,
                                                                    split="test",
                                                                    shuffle=False)

                feature_encoder_test.load_state_dict(torch.load('feature_encoder.txt'))
                relation_network_test.load_state_dict(torch.load('relation_network.txt'))
                reset_Encoder_Batchnorm(feature_encoder_test)

                sample_datas, sample_labels = sample_dataloader.__iter__().__next__()

                for adapt_epoch in range(adapt_epochs):
                    task1_sample_datas1, task1_sample_labels1 = task1_sample_dataloader.__iter__().__next__()
                    task1_sample_datas2, task1_sample_labels2 = task1_sample_dataloader.__iter__().__next__()
                    sample_datas1, task1_sample_datas1, task1_sample_datas2 = swap_dim(sample_datas, task1_sample_datas1, task1_sample_datas2)
                    task1_sample_datas1[:min(CLASS_NUM1,CLASS_NUM2) * SAMPLE_NUM_PER_CLASS,:,:] = task1_sample_datas1[:min(CLASS_NUM1,CLASS_NUM2) * SAMPLE_NUM_PER_CLASS,:,:] * 0.3 + sample_datas1[:min(CLASS_NUM1,CLASS_NUM2) * SAMPLE_NUM_PER_CLASS,:,:] * 0.7
                    task1_sample_datas2[:min(CLASS_NUM1,CLASS_NUM2) * SAMPLE_NUM_PER_CLASS,:,:] = task1_sample_datas2[:min(CLASS_NUM1,CLASS_NUM2) * SAMPLE_NUM_PER_CLASS,:,:] * 0.3 + sample_datas1[:min(CLASS_NUM1,CLASS_NUM2) * SAMPLE_NUM_PER_CLASS,:,:] * 0.7
                    sample_datas1,_ = perturb_dim(sample_datas1,sample_datas1)

                    sample_features = feature_encoder_test(
                        Variable(sample_datas1).permute(0, 2, 1).to(torch.float32).to(device))
                    task1_sample_features1 = feature_encoder_test(
                        Variable(task1_sample_datas1).permute(0, 2, 1).to(torch.float32).to(device))
                    task1_sample_features2 = feature_encoder_test(
                        Variable(task1_sample_datas2).permute(0, 2, 1).to(torch.float32).to(device))
                    sample_features, task1_sample_features1, task1_sample_features2 = swap_dim(sample_features, task1_sample_features1, task1_sample_features2)

                    sample_no = list(np.random.choice(np.arange(SAMPLE_NUM_PER_CLASS), size=2, replace=False)) + \
                                list(np.random.choice(np.arange(SAMPLE_NUM_PER_CLASS), size=2, replace=False)) + \
                                list(np.random.choice(np.arange(SAMPLE_NUM_PER_CLASS), size=2, replace=False)) + \
                                list(np.random.choice(np.arange(SAMPLE_NUM_PER_CLASS), size=2, replace=False)) + \
                                list(np.random.choice(np.arange(SAMPLE_NUM_PER_CLASS), size=2, replace=False)) + \
                                list(np.random.choice(np.arange(SAMPLE_NUM_PER_CLASS), size=2, replace=False))

                    sample_features_0 = sample_features[
                                        [i * SAMPLE_NUM_PER_CLASS + sample_no[0 + 2 * i] for i in range(CLASS_NUM2)], :,
                                        :]
                    sample_features_1 = sample_features[
                                        [i * SAMPLE_NUM_PER_CLASS + sample_no[1 + 2 * i] for i in range(CLASS_NUM2)], :,
                                        :]
                    sample_features_0_ext = sample_features_0.unsqueeze(0).repeat(CLASS_NUM2, 1, 1,
                                                                                  1)  # [[abcd][abcd][abcd][abcd]]
                    sample_features_1_ext = sample_features_1.unsqueeze(0).repeat(CLASS_NUM2, 1, 1, 1)
                    sample_features_1_ext = torch.transpose(sample_features_1_ext, 0, 1)  # [[aaaa][bbbb][cccc][dddd]]
                    relation_pairs1 = torch.cat((sample_features_0_ext, sample_features_1_ext), 2).view(-1,
                                                                                                        FEATURE_DIM * 2,
                                                                                                        LENGTH)
                    #print( [i * SAMPLE_NUM_PER_CLASS + sample_no[0 + 2 * i] for i in range(CLASS_NUM2)])
                    #print(sample_features.shape)
                    sample_features_0 = task1_sample_features1[
                                        [i * SAMPLE_NUM_PER_CLASS + sample_no[0 + 2 * i] for i in range(CLASS_NUM1)], :,
                                        :]
                    if CLASS_NUM1 < CLASS_NUM2:
                        sample_features_0 = torch.cat([sample_features_0,sample_features[5 * SAMPLE_NUM_PER_CLASS + sample_no[0 + 2 * 5]].unsqueeze(0)],0)
                    elif CLASS_NUM1 > CLASS_NUM2:
                        sample_features_0 = sample_features_0[:CLASS_NUM2,:,:]
                    sample_features_1 = sample_features[
                                        [i * SAMPLE_NUM_PER_CLASS + sample_no[1 + 2 * i] for i in range(CLASS_NUM2)], :,
                                        :]
                    sample_features_0_ext = sample_features_0.unsqueeze(0).repeat(CLASS_NUM2, 1, 1,
                                                                                  1)  # [[abcd][abcd][abcd][abcd]]
                    sample_features_1_ext = sample_features_1.unsqueeze(0).repeat(CLASS_NUM2, 1, 1, 1)
                    sample_features_1_ext = torch.transpose(sample_features_1_ext, 0, 1)  # [[aaaa][bbbb][cccc][dddd]]
                    relation_pairs2 = torch.cat((sample_features_0_ext, sample_features_1_ext), 2).view(-1,
                                                                                                        FEATURE_DIM * 2,
                                                                                                        LENGTH)

                    sample_features_0 = sample_features[
                                        [i * SAMPLE_NUM_PER_CLASS + sample_no[0 + 2 * i] for i in range(CLASS_NUM2)], :,
                                        :]
                    sample_features_1 = task1_sample_features2[
                                        [i * SAMPLE_NUM_PER_CLASS + sample_no[1 + 2 * i] for i in range(CLASS_NUM1)], :,
                                        :]
                    if CLASS_NUM1 < CLASS_NUM2:
                        sample_features_1 = torch.cat([sample_features_1,sample_features[5 * SAMPLE_NUM_PER_CLASS + sample_no[1 + 2 * 5]].unsqueeze(0)],0)
                    elif CLASS_NUM1 > CLASS_NUM2:
                        sample_features_1 = sample_features_1[:CLASS_NUM2,:,:]
                    sample_features_0_ext = sample_features_0.unsqueeze(0).repeat(CLASS_NUM2, 1, 1,
                                                                                  1)  # [[abcd][abcd][abcd][abcd]]
                    sample_features_1_ext = sample_features_1.unsqueeze(0).repeat(CLASS_NUM2, 1, 1, 1)
                    sample_features_1_ext = torch.transpose(sample_features_1_ext, 0, 1)  # [[aaaa][bbbb][cccc][dddd]]
                    relation_pairs3 = torch.cat((sample_features_0_ext, sample_features_1_ext), 2).view(-1,
                                                                                                        FEATURE_DIM * 2,
                                                                                                        LENGTH)

                    relation_pairs = torch.cat([relation_pairs1, relation_pairs2, relation_pairs3], 0)
                    temp_shape = relation_pairs.shape;temp_tensor = torch.zeros([temp_shape[0], 1, 8]).to(device)
                    relation_pairs = torch.cat((temp_tensor, relation_pairs), 1)

                    relations = relation_network_test(relation_pairs).view(-1, CLASS_NUM2)
                    real_labels = torch.cat(
                        [torch.tensor([i for i in range(CLASS_NUM2)]), torch.tensor([i for i in range(CLASS_NUM2)]),
                         torch.tensor([i for i in range(CLASS_NUM2)])], 0)
                    one_hot_labels = Variable(
                        torch.zeros(CLASS_NUM2 * 3, CLASS_NUM2).scatter_(1, real_labels.view(-1, 1),
                                                                         1).to(device))
                    loss = mse(relations, one_hot_labels)
                    feature_encoder_test.zero_grad()
                    relation_network_test.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(feature_encoder_test.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(relation_network_test.parameters(), 0.5)
                    feature_encoder_test_optim.step()
                    relation_network_test_optim.step()
                # reset_Encoder_Batchnorm(feature_encoder_test)
                # reset_Relation_Batchnorm(relation_network_test)
                for test_datas, test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = feature_encoder_test(
                        Variable(sample_datas).permute(0, 2, 1).to(torch.float32).to(device))
                    sample_features = sample_features.view(CLASS_NUM2, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, LENGTH)
                    sample_features = torch.mean(sample_features, 1).squeeze(1)
                    test_features = feature_encoder_test(
                        Variable(test_datas).permute(0, 2, 1).to(torch.float32).to(device))  # 20x64

                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    # to form a 100x128 matrix for relation network
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1)

                    test_features_ext = test_features.unsqueeze(0).repeat(1 * CLASS_NUM2, 1, 1, 1)

                    test_features_ext = torch.transpose(test_features_ext, 0, 1)

                    relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2,
                                                                                                 LENGTH)

                    temp_shape = relation_pairs.shape;
                    temp_tensor = torch.zeros([temp_shape[0], 1, 8]).to(device)
                    relation_pairs = torch.cat((temp_tensor, relation_pairs), 1)

                    relations = relation_network_test(relation_pairs).view(-1, CLASS_NUM2)

                    _, predict_labels = torch.max(relations.data, 1)

                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                    tp0 = [1 if predict_labels[j] == test_labels[j] and test_labels[j] == 0 else 0 for j in
                           range(batch_size)]
                    tp1 = [1 if predict_labels[j] == test_labels[j] and test_labels[j] == 1 else 0 for j in
                           range(batch_size)]
                    tp2 = [1 if predict_labels[j] == test_labels[j] and test_labels[j] == 2 else 0 for j in
                           range(batch_size)]
                    tp3 = [1 if predict_labels[j] == test_labels[j] and test_labels[j] == 3 else 0 for j in
                           range(batch_size)]
                    fn0 = [1 if predict_labels[j] != test_labels[j] and test_labels[j] == 0 else 0 for j in
                           range(batch_size)]
                    fn1 = [1 if predict_labels[j] != test_labels[j] and test_labels[j] == 1 else 0 for j in
                           range(batch_size)]
                    fn2 = [1 if predict_labels[j] != test_labels[j] and test_labels[j] == 2 else 0 for j in
                           range(batch_size)]
                    fn3 = [1 if predict_labels[j] != test_labels[j] and test_labels[j] == 3 else 0 for j in
                           range(batch_size)]
                    fp0 = [1 if predict_labels[j] != test_labels[j] and predict_labels[j] == 0 else 0 for j in
                           range(batch_size)]
                    fp1 = [1 if predict_labels[j] != test_labels[j] and predict_labels[j] == 1 else 0 for j in
                           range(batch_size)]
                    fp2 = [1 if predict_labels[j] != test_labels[j] and predict_labels[j] == 2 else 0 for j in
                           range(batch_size)]
                    fp3 = [1 if predict_labels[j] != test_labels[j] and predict_labels[j] == 3 else 0 for j in
                           range(batch_size)]
                    total_tp0 += np.sum(tp0);
                    total_tp1 += np.sum(tp1);
                    total_tp2 += np.sum(tp2);
                    total_tp3 += np.sum(tp3)
                    total_fn0 += np.sum(fn0);
                    total_fn1 += np.sum(fn1);
                    total_fn2 += np.sum(fn2);
                    total_fn3 += np.sum(fn3)
                    total_fp0 += np.sum(fp0);
                    total_fp1 += np.sum(fp1);
                    total_fp2 += np.sum(fp2);
                    total_fp3 += np.sum(fp3)
                    total_rewards += np.sum(rewards)
                    # print(tp0,fn0,fp0)
                accuracy = total_rewards / 1.0 / CLASS_NUM2 / BATCH_NUM_PER_CLASS2
                accuracies.append(accuracy)
                r0 = total_tp0 / (total_tp0 + total_fn0);
                r1 = total_tp1 / (total_tp1 + total_fn1);
                r2 = total_tp2 / (total_tp2 + total_fn2);
                r3 = total_tp3 / (total_tp3 + total_fn3);
                p0 = total_tp0 / (total_tp0 + total_fp0);
                p1 = total_tp1 / (total_tp1 + total_fp1);
                p2 = total_tp2 / (total_tp2 + total_fp2);
                p3 = total_tp3 / (total_tp3 + total_fp3);
                # print('r0:',r0,'--r1:',r1,'--r2:',r2,'--r3:',r3,'--p0:',p0,'--p1:',p1,'--p2:',p2,'--p3:',p3)
                f1_0 = 2 * r0 * p0 / (r0 + p0) if total_tp0 != 0 else 0
                f1_1 = 2 * r1 * p1 / (r1 + p1) if total_tp1 != 0 else 0
                f1_2 = 2 * r2 * p2 / (r2 + p2) if total_tp2 != 0 else 0
                f1_3 = 2 * r3 * p3 / (r3 + p3) if total_tp3 != 0 else 0
                f1 = (f1_0 + f1_1 + f1_2 + f1_3) / 4
                f1_all.append(f1)
            test_accuracy = np.mean(accuracies)
            test_f1 = np.mean(f1_all)
            accuracy_list.append(test_accuracy)
            f1_list.append(test_f1)
            feature_encoder.load_state_dict(torch.load('feature_encoder.txt'))
            relation_network.load_state_dict(torch.load('relation_network.txt'))
    print(accuracy_list)
    print(f1_list)
    mean_accuracy,h_accuracy = mean_confidence_interval(accuracy_list)
    mean_f1,h_f1 = mean_confidence_interval(f1_list)
    print(mean_accuracy)
    print(h_accuracy)
    print(mean_f1)
    print(h_f1)
if __name__ == '__main__':
    main()