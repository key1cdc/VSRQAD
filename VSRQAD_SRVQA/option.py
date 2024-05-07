import argparse
parser = argparse.ArgumentParser(description='Pre-trained image restoration')


# hardware setting
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--seed', type=int, default=56, help='random seed to use. Default=123')


#  train option
parser.add_argument('--patch_size', type=int, default=320, help='Ground truth patch size')
# parser.add_argument("--npy", action='store_true', help='choose npy as reader to reduce the loading time of images')
parser.add_argument("--npy", type=int,default=0, help='choose npy as reader to reduce the loading time of images')
parser.add_argument("--upscale", type=int,default=1, help='choose upscale')

parser.add_argument('--n_threads', type=int, default=4,
                    help='number of threads for data loading')
parser.add_argument("--nEpochs", type=int, default=1000, help="Number of epochs to train for")
parser.add_argument('--print_every', type=int, default=500,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_every', type=int, default=1,
                    help='how many batches to wait before saving training images')
parser.add_argument('--save_epoch', type=int, default=1,
                    help='how many epochs to wait before saving model and evaluating')



# Data specifications
parser.add_argument('--train_root', default=r'K:/VSR_613')
parser.add_argument('--test_root', default=r'K:/VSR_613')
parser.add_argument('--task', type=str, default='SR',
                    help='Task of our model. e.g. SR/CAR')
parser.add_argument('--data_reset', action='store_true',
                    help='Save images as .npy')
# parser.add_argument('--noise_type', type=str, default='G',
#                     help='Type of noise e.g. gaussian or poisson ')
parser.add_argument('--scale_param', type=int, default=4, help='Degradation scale. e.g. 2 for sr')
# parser.add_argument("--augment", action='store_true')
parser.add_argument("--augment", type=int, default=1)
parser.add_argument('--n_colors', type=int, default=3, help='Input image colors channel numbers')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')

# Optimization specifications
parser.add_argument('--lr', type=float, default=8e-4,
                    help='learning rate')
parser.add_argument('--lr2', type=float, default=3e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=50,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.85,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='SGD',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='L1',
                    help='loss function configuration e.g. L1+VGG or MSE+GAN')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Model specifications
parser.add_argument('--model', default='dreamSRVQA',
                    help='model name')

parser.add_argument('--gan_type', type=str, default='GAN', help='GAN type (GAN, LSGAN, WGANGP)')
parser.add_argument('--dis_type', type=str, default='pixel',
                    help='discriminator type (basic, patch, pixel)')

parser.add_argument('--frames_num', type=int, default=5, help='Input frames number')
# parser.add_argument('--n_resblocks', type=int, default=32, help='Residual block number')
parser.add_argument('--n_feats', type=int, default=64, help='Feats number')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--stage', type=str, default='train', help='Train or test stage')


parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
parser.add_argument('--n_resblocks', type=int, default=4, help='number of residual blocks')
parser.add_argument('--n_RFAs', type=int, default=32, help='number of residual blocks')
parser.add_argument('--reduction', type=int, default=8, help='number of feature maps reduction')
parser.add_argument('--scale', type=int, default=4, help='super resolution scale')

parser.add_argument('--train_phase', type=str, default='sr', help='Training phase: sr/fuse/joint')
parser.add_argument('--pre_train', type=str, default='.',
                    help='Pre-trained model name, it is different from resume')
parser.add_argument('--test_final', type=str, default='sr_epoch150')
parser.add_argument('--resume', type=int, default=0,     # if resume=0 then retrain; if resume>0, load model
                    help='resume from specific checkpoint')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

args = parser.parse_args()


for arg in vars(args):            # change str True to True
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

str_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:         # if id < 0; then apu_ids = []
        args.gpu_ids.append(id)

args.sr_factor = 0     # Shaving output and target in super resoltuion task when computing psnr.

if args.task == 'denoise':
    args.noise = [args.noise_type, args.scale_parame]
elif args.task == 'autoencoder':
    args.scale = 1
elif args.task == 'sr':
    args.sr_factor = args.scale
