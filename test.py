


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
            # Input
            save_image(images[ind].squeeze(0), os.path.join(result_path + str(ind) + '_' + str(i) + '_input' + '.png'))
            # PREDICTIONS
            save_image(output_fra1[ind].squeeze(0),
                       os.path.join(result_path + str(ind) + '_' + str(i) + '_pred_fra1' + '.png'))
            save_image(output_fra2[ind].squeeze(0),
                       os.path.join(result_path + str(ind) + '_' + str(i) + '_pred_fra2' + '.png'))
            save_image(output_lpw[ind].squeeze(0),
                       os.path.join(result_path + str(ind) + '_' + str(i) + '_pred_lpw' + '.png'))

            # TARGETS
            save_image(labels_fra1[ind].squeeze(0),
                       os.path.join(result_path + str(ind) + '_' + str(i) + '_target_fra1' + '.png'))
            save_image(labels_fra2[ind].squeeze(0),
                       os.path.join(result_path + str(ind) + '_' + str(i) + '_target_fra2' + '.png'))
            save_image(labels_lpw[ind].squeeze(0),
                       os.path.join(result_path + str(ind) + '_' + str(i) + '_target_lpw' + '.png'))

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
