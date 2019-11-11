"""Calculate precision on the seen classes of COCO."""

import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import shutil
from collections import defaultdict
from shutil import copyfile

from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm

from sklearn.metrics import precision_recall_curve, recall_score
from sklearn.metrics import average_precision_score

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from oneshot import setops_models
from oneshot import alfassy
from oneshot.coco import copy_coco_data

from experiment import Experiment

# from CCC import setupCUDAdevice

# setupCUDAdevice()

from ignite._utils import convert_tensor

from traitlets import Bool, Float, Int, Unicode

# setupCUDAdevice()
from scripts_coco.train_setops_stripped_new import FlagDatasetPairs, FLAG_CLASS_NUM, load_image

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def _prepare_batch(batch, device=None):
    return [convert_tensor(x, device=device) for x in batch]


class FlagDataset(Dataset):

    def __init__(self, root_dir, set_name, dataset_size_ratio=1, transform=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.dataset_size_ratio = dataset_size_ratio

        self.class_histogram = np.zeros(FLAG_CLASS_NUM)
        self.transform = transform

        self.list_id_img = []
        self.list_label_img = []
        self.id_img_to_labels = {}

        self.labels_to_img_ids = None
        self.labels_list = None

        logging.info("Calculating indices.")
        self.images_indices = None

        self.calc_indices()

    def get_path(self, idx):
        return os.path.join(self.root_dir, "images", self.set_name, self.get_img_name(idx))

    def __len__(self):
        return len(self.images_indices)

    def calc_indices(self):
        sum_ids = 0

        self.class_histogram = np.zeros(FLAG_CLASS_NUM)
        self.list_id_img = []
        self.id_img_to_labels = {}

        self.list_label_img = []
        labels = open(os.path.join(self.root_dir, self.set_name + '.txt'), 'r')
        line = labels.readline()
        pos_jpg = line.find('.jpg')
        self.images_indices = []
        self.labels_to_img_ids = defaultdict(list)
        while line:
            name_img = line[:pos_jpg]
            self.list_id_img.append(name_img)

            label_img_line = line[pos_jpg + 5:]
            label_img = np.array(list(map(int, label_img_line.split(" "))))

            self.list_label_img.append(label_img)

            line = labels.readline()
            self.images_indices.append((name_img, self.get_path(name_img), label_img))
            # for label_int in self.get_label_int(label_img):
            #     self.labels_to_img_ids[label_int].append(name_img)
        labels.close()

        self.labels_list = list(range(FLAG_CLASS_NUM))

    def __getitem__(self, idx):
        name_img, path, labels = self.images_indices[idx]
        img = load_image(path)
        if self.transform:
            img = self.transform(img)
        return name_img, img, labels

    def get_label_int(self, label_img):
        return np.where(label_img == 1)

    def get_img_name(self, name_img):
        return str(name_img) + ".jpg"

    def copy_label(self, label_int: int, target_folder):
        for name_img in self.labels_to_img_ids[label_int]:
            fr = self.get_path(name_img)
            to = os.path.join(target_folder,self.get_img_name(name_img))

            logging.info("Copying {} to {}".format(fr, to))
            shutil.copyfile(fr,to)



class Main(Experiment):
    description = Unicode(u"Calculate precision-recall accuracy of trained coco model.")

    #
    # Run setup
    #
    batch_size = Int(256, config=True, help="Batch size. default: 256")
    num_workers = Int(8, config=True, help="Number of workers to use for data loading. default: 8")
    if torch.cuda.is_available():
        device = Unicode("cuda", config=True, help="Use `cuda` backend.")
    else:
        device = DEVICE

    #
    # Hyper parameters.
    #
    unseen = Bool(False, config=True, help="Test on unseen classes.")
    skip_tests = Int(1, config=True, help="How many test pairs to skip? for better runtime. default: 1")
    debug_size = Int(-1, config=True,
                     help="Reduce dataset sizes. This is useful when developing the script. default -1")

    #
    # Resume previous run parameters.
    #
    resume_path = Unicode(u"/dccstor/alfassy/finalLaSO/code_release/paperModels", config=True,
                          help="Resume from checkpoint file (requires using also '--resume_epoch'.")
    resume_epoch = Int(0, config=True, help="Epoch to resume (requires using also '--resume_path'.")
    coco_path = Unicode(u"/tmp/aa/coco", config=True, help="path to local coco dataset path")
    init_inception = Bool(True, config=True, help="Initialize the inception networks using the paper's base network.")

    #
    # Network hyper parameters
    #
    base_network_name = Unicode("Inception3", config=True, help="Name of base network to use.")
    avgpool_kernel = Int(10, config=True,
                         help="Size of the last avgpool layer in the Resnet. Should match the cropsize.")
    classifier_name = Unicode("Inception3Classifier", config=True, help="Name of classifier to use.")
    sets_network_name = Unicode("SetOpsResModule", config=True, help="Name of setops module to use.")
    sets_block_name = Unicode("SetopResBlock_v1", config=True, help="Name of setops network to use.")
    sets_basic_block_name = Unicode("SetopResBasicBlock", config=True,
                                    help="Name of the basic setops block to use (where applicable).")
    ops_layer_num = Int(1, config=True, help="Ops Module layers num.")
    ops_latent_dim = Int(1024, config=True, help="Ops Module inner latent dim.")
    setops_dropout = Float(0, config=True, help="Dropout ratio of setops module.")
    crop_size = Int(299, config=True, help="Size of input crop (Resnet 224, inception 299).")
    scale_size = Int(350, config=True, help="Size of input scale for data augmentation. default: 350")
    paper_reproduce = Bool(False, config=True, help="Use paper reproduction settings. default: False")
    discriminator_name = Unicode("AmitDiscriminator", config=True,
                                 help="Name of discriminator (unseen classifier) to use. default: AmitDiscriminator")
    embedding_dim = Int(2048, config=True, help="Dimensionality of the LaSO space. default:2048")
    classifier_latent_dim = Int(2048, config=True, help="Dimensionality of the classifier latent space. default:2048")

    results_path = Unicode("", config=True, help="Base path for experiment results.")
    num_class = Int(80, config=True, help="")

    def run(self):

        #
        # Setup the model
        #
        base_model, classifier, setops_model = self.setup_model(self.num_class)

        base_model.to(self.device)
        classifier.to(self.device)
        setops_model.to(self.device)

        base_model.eval()
        classifier.eval()
        setops_model.eval()

        #
        # Load the dataset
        #
        # pair_dataset, pair_loader, pair_dataset_sub, pair_loader_sub = self.setup_datasets()
        dataset = self.setup_datasets()

        logging.info("Calcualting classifications:")
        output_list = []
        target_list = []
        copy_folder_target = "/home/nganltp/laso/data/flags/images/false/"
        total_false = 0
        # predict_file = open("/home/nganltp/laso/data/flags/images/predict.txt", mode='w+')
        TP = np.zeros(7)
        FN = np.zeros(7)
        FP = np.zeros(7)
        TN = np.zeros(7)
        with torch.no_grad():
            for name_img, inp, target in tqdm(dataset):
                inp = inp.unsqueeze(0)
                target = torch.tensor(target).to(DEVICE)
                target = target.unsqueeze(0)
                #
                # Apply the classification model
                #
                embed = base_model(inp).view(inp.size(0), -1)
                output = classifier(embed)
                #
                # Apply the setops model.
                #

                output_list.append(output.cpu().numpy())

                sig_out = (torch.sigmoid(output) > 0.5).int()
                # logging.info("Image {}, label {}, predicted {}".format(name_img, target, sig_out))
                #
                # Calculate the target setops operations
                #
                target_list.append(target.cpu().numpy())
                sig_out = sig_out.squeeze(0)
                v_target=target[0]
# ------------------------------confusion matrix--------------------------------------------
                for lab in range(7):
                    if v_target[lab] == 1 and sig_out[lab] == 1:
                        TP[lab] += 1

                    if v_target[lab] == 1 and sig_out[lab] == 0:
                        FN[lab] += 1

                    if v_target[lab] == 0 and sig_out[lab] == 1:
                        FP[lab] += 1

                    if v_target[lab] == 0 and sig_out[lab] == 0:
                        TN[lab] += 1

                logging.info('TP: {}, FN: {}, FP: {}, TN: {}'.format(TP, FN, FP, TN))
# ------------------------------------------------------------------------------------------
                # if torch.sum(target) == 0 and (sig_out[1] == 1):#or sig_out[5] == 1 or sig_out[6] == 1):
                # logging.info('type target {}, sig_out {}'.format(type(target), type(sig_out)))
                # if ((target[0][1] == 0) and (sig_out[1] == 1)) or ((target[0][1] == 1) and (sig_out[1] == 0)):
                # org_path = dataset.get_path(name_img)
                # copyfile(org_path, copy_folder_target + str(name_img) + "_" + str(sig_out) + "_" + ".jpg")
                # total_false += 1
                # logging.info("Image {}, label {}, predicted {}".format(name_img, target, sig_out))
                # predict_file.write('Image: ' + str(name_img) + ', target: ' + str(target) + ',predicted: ' + str(sig_out) + '\n')

        logging.info("Total false {}".format(total_false))
        output = np.array(output_list).squeeze(1)
        target = np.array(target_list).squeeze(1)

        ap = [average_precision_score(target[:, i], output[:, i]) for i in range(output.shape[1])]
        # recall = [recall_score(target[:, i], output[:, i] ) for i in range(output.shape[1])]
        pr_graphs = [precision_recall_curve(target[:, i], output[:, i]) for i in range(output.shape[1])]
        ap_sum = 0
        recall_sum = 0
        logging.info('class precision {}'.format(ap))
        # logging.info('class recall{}'.format(recall))
        for label in dataset.labels_list:
            ap_sum += ap[label]
            # recall_sum += recall[label]
        ap_avg = ap_sum / len(dataset.labels_list)
        # recall_avg = recall_sum / len(dataset.labels_list)
        logging.info(
            'Test {} average precision score, macro-averaged over all {} classes: {}'.format(
                "Single Image test score", len(dataset.labels_list), ap_avg)
        )
        # logging.info(
        #     'Test {} average recall score, macro-averaged over all {} classes: {}'.format(
        #         "Single Image test score", len(dataset.labels_list), recall_avg)
        # )

    def setup_model(self, num_classes=80):
        """Create or resume the models."""

        logging.info("Setup the models.")

        logging.info("{} model".format(self.base_network_name))
        models_path = Path(self.resume_path)
        if self.base_network_name.lower().startswith("resnet"):
            base_model, classifier = getattr(setops_models, self.base_network_name)(
                num_classes=num_classes,
                avgpool_kernel=self.avgpool_kernel
            )
        else:
            base_model = setops_models.Inception3(aux_logits=False, transform_input=True)
            classifier = getattr(setops_models, self.classifier_name)(num_classes=80)
            if self.init_inception:
                logging.info("Initialize inception model using paper's networks.")
                checkpoint = torch.load(models_path / 'paperBaseModel')
                base_model = setops_models.Inception3(aux_logits=False, transform_input=True)
                base_model.load_state_dict(
                    {k: v for k, v in checkpoint["state_dict"].items() if k in base_model.state_dict()}
                )
                classifier.load_state_dict(
                    {k: v for k, v in checkpoint["state_dict"].items() if k in classifier.state_dict()}
                )
        setops_model_cls = getattr(setops_models, self.sets_network_name)
        setops_model = setops_model_cls(
            input_dim=self.embedding_dim,
            S_latent_dim=self.ops_latent_dim, S_layers_num=self.ops_layer_num,
            I_latent_dim=self.ops_latent_dim, I_layers_num=self.ops_layer_num,
            U_latent_dim=self.ops_latent_dim, U_layers_num=self.ops_layer_num,
            block_cls_name=self.sets_block_name, basic_block_cls_name=self.sets_basic_block_name,
            dropout_ratio=self.setops_dropout,
        )

        if self.unseen:
            #
            # In the unseen mode, we have to load the trained discriminator.
            #
            discriminator_cls = getattr(setops_models, self.discriminator_name)
            classifier = discriminator_cls(
                input_dim=self.embedding_dim,
                latent_dim=self.classifier_latent_dim
            )

        if not self.resume_path:
            raise FileNotFoundError("resume_path is compulsory in test_precision")
        logging.info("Resuming the models.")
        if not self.init_inception:
            base_model.load_state_dict(
                torch.load(sorted(models_path.glob("networks_base_model_{}*.pth".format(self.resume_epoch)))[-1],
                           map_location=DEVICE)
            )

        if self.paper_reproduce:
            logging.info("using paper models")
            setops_model_cls = getattr(setops_models, "SetOpsModulePaper")
            setops_model = setops_model_cls(models_path)
            if self.unseen:
                checkpoint = torch.load(models_path / 'paperDiscriminator')
                classifier.load_state_dict(checkpoint['state_dict'])
        else:
            setops_model.load_state_dict(
                torch.load(
                    sorted(
                        models_path.glob("networks_setops_model_{}*.pth".format(self.resume_epoch))
                    )[-1]
                    , map_location=DEVICE)
            )
            if self.unseen:
                classifier.load_state_dict(
                    torch.load(sorted(models_path.glob("networks_discriminator_{}*.pth".format(self.resume_epoch)))[-1])
                )
            elif not self.init_inception:
                classifier.load_state_dict(
                    torch.load(sorted(models_path.glob("networks_classifier_{}*.pth".format(self.resume_epoch)))[-1],
                               map_location=DEVICE)
                )

        return base_model, classifier, setops_model

    def setup_datasets(self):
        """Load the training datasets."""
        # TODO: comment out if you don't want to copy coco to /tmp/aa
        # copy_coco_data()

        logging.info("Setting up the datasets.")
        if self.paper_reproduce:
            logging.info("Setting up the datasets and augmentation for paper reproduction")
            scaler = transforms.Scale((350, 350))
        else:
            scaler = transforms.Resize(self.crop_size)

        val_transform = transforms.Compose(
            [
                scaler,
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        load_dataset = FlagDataset(
            root_dir=self.coco_path,
            set_name='val',
            transform=val_transform,
        )

        # pair_loader = DataLoader(
        #     pair_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers
        # )
        # pair_dataset_sub = FlagDatasetPairs(
        #     root_dir=self.coco_path,
        #     set_name='val',
        #     transform=val_transform,
        # )
        #
        # pair_loader_sub = DataLoader(
        #     pair_dataset_sub,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers
        # )

        return load_dataset


if __name__ == "__main__":
    main = Main()
    main.initialize()
    main.start()
