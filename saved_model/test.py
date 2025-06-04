import torch
if __name__ == '__main__':

    checkpoint_path = "saved_model/checkpoint-best.pth"

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

    print(checkpoint.keys())
