import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from tqdm import tqdm
import argparse

from dataset import RESIDEOUTDataset
from model import MHCAFNetPP

def main():
    parser = argparse.ArgumentParser(description="Train MHCAFNet++")
    parser.add_argument("--data_dir", type=str, default=r"D:\David DLT\data\RESIDE OUT\RESIDE OUT", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Device:", device)

    # Dataset
    train_dataset = RESIDEOUTDataset(args.data_dir, "train")
    test_dataset  = RESIDEOUTDataset(args.data_dir, "test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = MHCAFNetPP().to(device)
    multi_gpu = torch.cuda.device_count() > 1
    if multi_gpu:
        model = nn.DataParallel(model)
        print(" utilisant DataParallel")
    else:
        print(f"🚀 Running on single GPU")

    # Loss and Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    perceptual = lpips.LPIPS(net='alex').to(device)

    best_psnr = 0
    counter = 0

    train_psnr_hist, test_psnr_hist = [], []
    train_ssim_hist, test_ssim_hist = [], []
    loss_hist = []

    print(f"🚀 Starting training on {device}...")

    for epoch in range(args.epochs):
        model.train()
        total_loss, t_psnr, t_ssim = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for hazy, gt, _ in loop:
            hazy, gt = hazy.to(device), gt.to(device)

            optimizer.zero_grad()
            pred = model(hazy)

            loss = (criterion(pred, gt)
                    + 0.2 * (1 - ssim(pred, gt))
                    + 0.1 * perceptual(pred, gt).mean())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            t_psnr += psnr(pred, gt).item()
            t_ssim += ssim(pred, gt).item()

            loop.set_postfix(loss=loss.item())

        train_psnr = t_psnr / len(train_loader)
        train_ssim = t_ssim / len(train_loader)
        train_psnr_hist.append(train_psnr)
        train_ssim_hist.append(train_ssim)
        loss_hist.append(total_loss / len(train_loader))

        # Validation
        model.eval()
        te_psnr, te_ssim = 0, 0
        with torch.no_grad():
            for hazy, gt, _ in test_loader:
                hazy, gt = hazy.to(device), gt.to(device)
                pred = model(hazy)
                te_psnr += psnr(pred, gt).item()
                te_ssim += ssim(pred, gt).item()

        test_psnr = te_psnr / len(test_loader)
        test_ssim = te_ssim / len(test_loader)
        test_psnr_hist.append(test_psnr)
        test_ssim_hist.append(test_ssim)

        print(f"\n📊 Summary Epoch {epoch+1}")
        print(f"Train -> Loss: {loss_hist[-1]:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
        print(f"Test  -> PSNR: {test_psnr:.2f}, SSIM: {test_ssim:.4f}")

        # SAVE LOCAL CHECKPOINT
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

        if test_psnr > best_psnr:
            best_psnr = test_psnr
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"⭐ New Best Model Saved (PSNR: {best_psnr:.2f})")
        else:
            counter += 1
            print(f"⚠️ No improvement. Early stopping counter: {counter}/{args.patience}")
            if counter >= args.patience:
                print("🛑 Early stopping triggered. Training halted.")
                break

    # Final Save
    save_path = "MHCAFNetPP_RESIDE_Final.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_hist[-1],
        'psnr': test_psnr_hist[-1]
    }, save_path)
    print(f"✅ Final model and training state saved successfully to: {save_path}")

if __name__ == '__main__':
    main()
