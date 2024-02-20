import einops

from .param_checking import to_2tuple


def get_sequence_lengths(input_shape, patch_size):
    assert len(input_shape) == len(patch_size)
    ndim = len(patch_size)
    assert all(input_shape[i] % patch_size[i] == 0 for i in range(ndim))
    seqlens = [input_shape[i] // patch_size[i] for i in range(ndim)]
    return seqlens


def sequence_to_2d_with_seqlens(tokens, h_seqlen, w_seqlen, num_aux_tokens):
    # transform into image with c=feature_dim h=h_seqlen w=w_seqlen
    aux_tokens = tokens[:, :num_aux_tokens]
    patch_tokens = tokens[:, num_aux_tokens:]
    img = einops.rearrange(
        patch_tokens,
        "b (h_seqlen w_seqlen) c -> b c h_seqlen w_seqlen",
        h_seqlen=h_seqlen,
        w_seqlen=w_seqlen,
    )
    return img, aux_tokens

def patchify_as_1d(x, patch_size):
    assert x.ndim - 2 == len(patch_size)
    ndim = len(patch_size)
    resolution = x.shape[2:]
    assert all(resolution[i] % patch_size[i] == 0 for i in range(ndim))
    seqlens = [resolution[i] // patch_size[i] for i in range(ndim)]
    # generate generic pattern for ndim
    # pattern for 2d is: "bs c (h ph) (w pw) -> bs (h w) (ph pw c)"
    # pattern for 3d is: "bs c (x px) (y py) (z pz) -> bs (x y z) (px py pz c)"
    from_pattern = "c " + " ".join([f"(seqlen{i} patchsize{i})" for i in range(ndim)])
    to_pattern1 = " ".join([f"seqlen{i}" for i in range(ndim)])
    to_pattern2 = " ".join([f"patchsize{i}" for i in range(ndim)]) + " c"
    kwargs = {f"seqlen{i}": seqlens[i] for i in range(ndim)}
    x = einops.rearrange(x, f"bs {from_pattern} -> bs ({to_pattern1}) ({to_pattern2})", **kwargs)
    return x


def patchify_as_2d(imgs, patch_size):
    patch_height, patch_width = to_2tuple(patch_size)
    bs, c, img_h, img_w = imgs.shape
    assert img_h % patch_height == 0 and img_w % patch_width == 0
    # how many patches are along height/width dimension
    h = img_h // patch_height
    w = img_w // patch_width
    # return as "image"
    x = einops.rearrange(imgs, "bs c (h ph) (w pw) -> bs (ph pw c) h w", h=h, ph=patch_height, w=w, pw=patch_width)
    return x


def unpatchify(patches, patch_size, img_shape=None):
    if patches.ndim == 3:
        return unpatchify_from_1d(patches=patches, patch_size=patch_size, img_shape=img_shape)
    elif patches.ndim == 4:
        if patches.shape[1:] == img_shape:
            return patches
        return unpatchify_from_2d(patches=patches, patch_size=patch_size)
    raise NotImplementedError


def unpatchify_from_1d(patches, patch_size, img_shape=None):
    remove_channel_dim = False
    assert patches.ndim == 3
    patch_height, patch_width = to_2tuple(patch_size)
    assert patch_height == patch_width or img_shape is not None
    if img_shape is not None:
        # derive number of patches along height/width from original image shape
        if len(img_shape) == 2:
            img_h, img_w = img_shape
            remove_channel_dim = True
        else:
            _, img_h, img_w = img_shape
        assert img_h % patch_height == 0 and img_w % patch_width == 0
        seqlen_h = img_h // patch_height
        seqlen_w = img_w // patch_width
    else:
        # equal number of patches along height/width
        seqlen_h = seqlen_w = int(patches.shape[1] ** .5)
    img = einops.rearrange(
        patches,
        "bs (seqlen_h seqlen_w) (ph pw c) -> bs c (seqlen_h ph) (seqlen_w pw)",
        ph=patch_height,
        pw=patch_width,
        seqlen_h=seqlen_h,
        seqlen_w=seqlen_w,
    )
    if remove_channel_dim:
        img = einops.rearrange(img, "bs 1 img_h img_w -> bs img_h img_w")
    return img


def unpatchify_from_2d(patches, patch_size):
    patch_height, patch_width = to_2tuple(patch_size)
    return einops.rearrange(patches, "bs (ph pw c) h w -> bs c (h ph) (w pw)", ph=patch_height, pw=patch_width)
