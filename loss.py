import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def bilinear_sampler_1d_h(image,disp):

    # shape
    B,C,H,W = image.shape

    # grid
    x = paddle.arange(W)
    y = paddle.arange(H)
    y_grid,x_grid = paddle.meshgrid(y,x)
    y_grid = y_grid.reshape((1,1,H,W)).expand((B,-1,-1,-1))
    x_grid = x_grid.reshape((1,1,H,W)).expand((B,-1,-1,-1)) + disp

    y_grid = 1.0*y_grid/(H-1.0)
    x_grid = 1.0*x_grid/(W-1.0)

    grid = paddle.concat([x_grid,y_grid],axis=1)
    grid = grid.transpose((0,2,3,1))

    return F.grid_sample(image,grid,mode='bilinear')

def generate_image_left(image,disp):
    return bilinear_sampler_1d_h(image, -disp)

def generate_image_right(image,disp):
    return bilinear_sampler_1d_h(image,disp)

def gradient_x(img):
    gx = img[:,:,:,:-1] - img[:,:,:,1:]
    return gx

def gradient_y(img):
    gy = img[:,:,:-1,:] - img[:,:,1:,:]
    return gy

def get_disparity_smoothness(image,disp):
    
    disp_grad_x = gradient_x(F.pad(disp,pad=[1,0,0,0]))
    disp_grad_y = gradient_y(F.pad(disp,pad=[0,0,1,0]))

    image_grad_x = gradient_x(F.pad(image,pad=[1,0,0,0]))
    image_grad_y = gradient_y(F.pad(image,pad=[0,0,1,0]))

    weight_x = paddle.exp(-image_grad_x.abs().mean())
    weight_y = paddle.exp(-image_grad_y.abs().mean())

    smooth_x = disp_grad_x*weight_x
    smooth_y = disp_grad_y*weight_y

    print('smooth_x',smooth_x.shape)
    print('smooth_y',smooth_y.shape)

    return smooth_x + smooth_y

def SSIM(x,y):
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = F.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return paddle.clip((1 - SSIM) / 2, 0, 1)

def total_loss(left_image,left_disps,right_image,right_disps,alpha_image_loss=1.0,disp_smooth_weight=1.0,disp_consis_weight=1.0 ):

    # build image pyramid
    left_images = []
    right_images = []
    pyramid_size = len(left_disps)
    for s in range(pyramid_size):
        left_images.append(left_image)
        right_images.append(right_image)

        left_image = nn.functional.upsample(left_image,scale_factor=0.5,mode='bilinear')    
        right_image = nn.functional.upsample(right_image,scale_factor=0.5,mode='bilinear')    

    # generate images
    left_images_est = []
    right_images_est = []
    for s in range(pyramid_size):
        left_image_est = generate_image_left(right_images[s],left_disps[s])
        right_image_est = generate_image_right(left_images[s],right_disps[s])

        left_images_est.append(left_image_est)
        right_images_est.append(right_image_est)

    # generate disparity
    left_to_right_disps = []
    right_to_left_disps = []
    for s in range(pyramid_size):
        left_to_right_disp = generate_image_right(left_disps[s],right_disps[s])
        right_to_left_disp = generate_image_left(right_disps[s],left_disps[s])    

        left_to_right_disps.append(left_to_right_disp)
        right_to_left_disps.append(right_to_left_disp)

    # disparity smoothness
    left_disps_smooth = []
    right_disps_smooth = []
    for s in range(pyramid_size):
        left_disp_smooth = get_disparity_smoothness(left_images[s],left_disps[s])
        right_disp_smooth = get_disparity_smoothness(right_images[s],right_disps[s])

        left_disps_smooth.append(left_disp_smooth)
        right_disps_smooth.append(right_disp_smooth)


    # image reconstruction 
    l1_reconstruction_loss = 0
    for s in range(pyramid_size):
        l1_reconstruction_loss += (left_images[s]-left_images_est[s]).abs().mean()
        l1_reconstruction_loss += (right_images[s]-right_images_est[s]).abs().mean()

    # ssim
    ssim_loss = 0
    for s in range(pyramid_size):
        ssim_loss += SSIM(left_images[s],left_images_est[s])
        ssim_loss += SSIM(right_images[s],right_images_est[s])
    
    # disparity smooth
    disp_smooth = 0
    for s in range(pyramid_size):
        disp_smooth += left_disps_smooth[s].abs().mean()/2**s
        disp_smooth += right_disps_smooth[s].abs().mean()/2**s

    # left right consistency
    disp_consistency = 0
    for s in range(pyramid_size):
        disp_consistency += (left_disps[s]-right_to_left_disps[s]).abs().mean()
        disp_consistency += (right_disps[s]-left_to_right_disps[s]).abs().mean()

    total_loss = 0
    total_loss += alpha_image_loss*ssim_loss + (1.0-alpha_image_loss)*l1_reconstruction_loss 
    total_loss += disp_smooth_weight*disp_smooth 
    total_loss += disp_consis_weight*disp_consistency

    return total_loss


if __name__ == '__main__':

    left_image = paddle.rand((1,1,256,512))
    right_image = paddle.rand((1,1,256,512))

    left_disp = paddle.rand((1,1,256,512))
    right_disp = paddle.rand((1,1,256,512))

    total_loss(left_image,[left_disp],right_image,[right_disp])