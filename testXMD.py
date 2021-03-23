
result_path = '/home/SyntheticResults/'
real_result_path = '/home/RealResults/'

# define the training function
def test(test_loader, model, criterion, device):
    batch_size = 4
    losses = AverageMeter()
    # Progress bar
    is_nan = []
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, (images, labels_fra1, labels_fra2, labels_lpw, index) in pbar:
        # Convert torch tensor to Variable
        images = images.to(device, dtype=torch.float)
        labels_fra1 = labels_fra1.to(device, dtype=torch.float)
        labels_fra2 = labels_fra2.to(device, dtype=torch.float)
        labels_lpw = labels_lpw.to(device, dtype=torch.float)

        # compute output
        optimizer.zero_grad()
        output_lpw, output_fra = model(images)
        output_fra1 = output_fra[:, 0, :, :].unsqueeze(1)
        output_fra2 = output_fra[:, 1, :, :].unsqueeze(1)

        # measure loss
        loss_fra1 = 1 - criterion(output_fra1, labels_fra1)
        loss_fra2 = 1 - criterion(output_fra2, labels_fra2)
        loss_lpw = 1 - criterion(output_lpw, labels_lpw)

        # Total losses
        recon = output_lpw - (1 - output_fra1) + (1 - output_fra2)
        loss_rec = 1 - criterion(recon, images)

        if np.isnan(loss_fra1.item()) or np.isnan(loss_fra2.item()) or np.isnan(loss_lpw.item()) or np.isnan(
                loss_rec.item()):
            print(index, i)
            sys.exit("nan")

        loss_tot = (loss_fra1 + loss_fra2 + 2 * loss_lpw + loss_rec) / 5
        loss_total = loss_tot.item()
        losses.update(loss_total, images.size(0))

        for ind in range(batch_size):
            im = torch.Tensor.cpu(images[ind].squeeze(0)).detach().numpy()
            label_lpw = torch.Tensor.cpu(output_lpw[ind].squeeze(0)).detach().numpy()
            fr1 = torch.Tensor.cpu(output_fra1[ind].squeeze(0)).detach().numpy()
            fr2 = torch.Tensor.cpu(output_fra2[ind].squeeze(0)).detach().numpy()
            img_color = np.array((im, fr1, fr2))

            img_color = np.transpose((img_color), (1, 2, 0)) * 255

            img_color1 = np.zeros((sz, sz, 3))
            img_color1[:, :, 0] = label_lpw * 255
            img_color2 = np.zeros((sz, sz, 3))
            img_color2[:, :, 1] = 2 * (255 - img_color[:, :, 1]) + label_lpw * 255
            img_color3 = np.zeros((sz, sz, 3))
            img_color3[:, :, 2] = 2 * (255 - img_color[:, :, 2]) + label_lpw * 255

            img_color_tot = np.zeros((sz, sz, 3))
            img_color_tot = (img_color1 + img_color2 + img_color3)

            ## Enhanced grayscale
            images_enh_n = .7 * (images[ind].squeeze(0)) - (1 - output_fra1[ind].squeeze(0)) + (
                        1 - output_fra2[ind].squeeze(0))
            images_enh_n = images_enh_n - torch.min(images_enh_n)
            images_enh_n = images_enh_n / torch.max(images_enh_n)
            save_image(images_enh_n, os.path.join(result_path + str(ind) + '_' + str(i) + '_enh_n' + '.png'))
            ## Resulted - color

    return losses.avg



class TRAILIID_real(Dataset):

    def __init__(self, root):
        """
        :param root: it has to be a path to the folder that contains the dataset folders

        """

        # initialize variables
        self.root = os.path.expanduser(root)
        self.path = []

        def load_images(path):
            images_dir = [join(path, f) for f in os.listdir(path) if isfile(join(path, f))]
            return images_dir

        self.path = load_images(self.root)

    def transform(self, _input):
        _input = TTF.to_grayscale(_input, 1)

        _input = TTF.resize(_input, (sz, sz), 2)
        _input = TTF.to_tensor(_input)

        _input -= torch.min(_input)
        if torch.max(_input) != 0:
            _input /= torch.max(_input)

        return _input

    def __getitem__(self, index):
        """

        :param index: image index
        :return: tuple (_def = img, _pwc = target) with the input data and its ground truth
        """

        _def = Image.open(self.path[index])
        _def = self.transform(_def)

        return _def

    def __len__(self):
        return len(self.path)


testrealDS = TRAILIID_real(root='/home/REAL_TEST/')
test_rloader = torch.utils.data.DataLoader(dataset=testrealDS,
                                           batch_size=4,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=1)
# define the training function
def test_real(test_rloader, model, criterion, device):
    losses = AverageMeter()
    batch_size = 4
    # Progress bar
    pbar = tqdm(enumerate(test_rloader), total=len(test_rloader))
    for i, (images) in pbar:
        # Converting torch tensor to Variable
        images = images.to(device, dtype=torch.float)

        # Computing the outputs
        optimizer.zero_grad()
        output_lpw, output_fra = model(images)
        output_fra1 = output_fra[:, 0, :, :].unsqueeze(1)
        output_fra2 = output_fra[:, 1, :, :].unsqueeze(1)

        for ind in range(images.shape[0]):
            # Input
            im = torch.Tensor.cpu(images[ind].squeeze(0)).detach().numpy()
            label_lpw = torch.Tensor.cpu(output_lpw[ind].squeeze(0)).detach().numpy()
            fr1 = torch.Tensor.cpu(output_fra1[ind].squeeze(0)).detach().numpy()
            fr2 = torch.Tensor.cpu(output_fra2[ind].squeeze(0)).detach().numpy()

            # RESULTS
            ## Enhanced grayscale
            images_enh_n = .7 * (images[ind].squeeze(0)) - (1 - output_fra1[ind].squeeze(0)) + (
                        1 - output_fra2[ind].squeeze(0))
            images_enh_n = images_enh_n - torch.min(images_enh_n)
            images_enh_n = images_enh_n / torch.max(images_enh_n)
            save_image(images_enh_n, os.path.join(real_result_path + str(ind) + '_' + str(i) + '_enh_n' + '.png'))

    return losses.avg
