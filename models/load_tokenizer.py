from models import tokenizer
import os
import torch


def load_pretrained_tokenizer(args):
    print("Loading pretrained tokenizer...")

    # -------------- build tokenizer ----------------
    encoder_config = tokenizer.get_model_default_params(args.signal_total_length, args.tokenizer_patch_size,args.num_channels)
    decoder_config = tokenizer.get_model_default_params(args.signal_total_length, args.tokenizer_patch_size,args.num_channels)

    # modify decoder parameters
    decoder_config['sig_size'] = encoder_config['sig_size'] // encoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['depth'] = 3
    decoder_config['embed_dim'] = 33 * (12 // args.num_channels) + 1
    decoder_config['patch_embed'] = tokenizer.PatchEmbed(
        EEG_size=decoder_config['sig_size'],
        patch_size=decoder_config['patch_size'],
        in_chans=decoder_config['in_chans'],
        embed_dim=decoder_config['embed_dim']
    )

    model = tokenizer.ECGTokenizer(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        n_embed=args.codebook_size,
        embed_dim=args.codebook_embed_dim,
        patch_size=args.tokenizer_patch_size,
        decoder_out_dim=args.tokenizer_patch_size  # Predict spectrum for each patch
    )

    # ------------- load weights ---------------
    if args.tokenizer_weight_path:
        if os.path.isfile(args.tokenizer_weight_path):
            print(f"Loading tokenizer weights from: {args.tokenizer_weight_path}")
            checkpoint = torch.load(args.tokenizer_weight_path, map_location='cpu')
            state_dict = checkpoint['model']
            model.load_state_dict(state_dict)
        else:
            print(f"ERROR: Tokenizer weight path not found: {args.tokenizer_weight_path}")
            exit(1)
    else:
        print("ERROR: No tokenizer weight path provided (--tokenizer_weight_path).")
        exit(1)

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print(f"Pretrained ECGTokenizer loaded and frozen.")
    return model.to(args.device)






