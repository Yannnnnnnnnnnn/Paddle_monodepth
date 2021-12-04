import numpy as np

import paddle
import paddle.nn as nn

class identiy(nn.Layer):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x):
        return x

class conv(nn.Layer):

    def __init__(self,in_channels,out_channels,kernel_size,padding,stride=1,active=True) -> None:
        super().__init__()

        self.conv = nn.Conv2D(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,stride=stride)

        self.active = active

        if active:
            self.elu  = nn.ELU()
    
    def forward(self,x):
        
        if self.active:
            return self.elu(self.conv(x))
        else:
            return self.conv(x)

class conv_block(nn.Layer):

    def __init__(self,in_channels,out_channels,kernel_size,padding) -> None:
        super().__init__()
    
        self.conv1 = conv(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,stride=1)
        self.conv2 = conv(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,stride=2)

    def forward(self,x):

        return self.conv2(self.conv1(x))

class upconv(nn.Layer):

    def __init__(self,in_channels,out_channels,kernel_size,padding) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = conv(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,stride=1)
    
    def forward(self,x):
        return self.conv(self.upsample(x))

class get_disp(nn.Layer):

    def __init__(self,in_channels,out_channels,kernel_size,padding) -> None:
        super().__init__()

        self.conv = conv(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,active=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        return self.sigmoid(self.conv(x)) * 0.3

class resconv(nn.Layer):

    def __init__(self,in_channels,out_channels,stride) -> None:
        super().__init__()

        self.conv1 = conv(in_channels=in_channels,out_channels=in_channels,kernel_size=1,padding=0,stride=1)
        self.conv2 = conv(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1,stride=stride)
        self.conv3 = conv(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0,stride=1,active=False)

        if stride==2 or in_channels!=out_channels:
            self.shortcut = conv(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=0,stride=stride,active=False)
        else:
            self.shortcut = identiy()

        self.elu = nn.ELU()

    def forward(self,x):
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        shortcut = self.shortcut(x)

        return self.elu(shortcut+out)

class resblock(nn.Layer):

    def __init__(self,in_channels,channels,num_blocks) -> None:
        super().__init__()

        self.layers = []
        for i in range(num_blocks):
            if i==0:
                layer = self.add_sublayer("",resconv(in_channels=in_channels,out_channels=4*channels,stride=1))
            elif i==(num_blocks-1):
                layer = self.add_sublayer("",resconv(in_channels=4*channels,out_channels=channels,stride=2))
            else:
                layer = self.add_sublayer("",resconv(in_channels=4*channels,out_channels=4*channels,stride=1))
            self.layers.append(layer)
    
    def forward(self,x):

        for layer in self.layers:
            x = layer(x)

        return x

class ResNet50(nn.Layer):

    def __init__(self,in_channels) -> None:
        super().__init__()
    
        self.conv1 = conv(in_channels=in_channels,out_channels=64,kernel_size=7,padding=3,stride=2)
        self.pool1 = nn.MaxPool2D(kernel_size=3,padding=1,stride=2)
        self.conv2 = resblock(in_channels=64,channels=64,num_blocks=3)
        self.conv3 = resblock(in_channels=64,channels=128,num_blocks=4)
        self.conv4 = resblock(in_channels=128,channels=256,num_blocks=6)
        self.conv5 = resblock(in_channels=256,channels=512,num_blocks=3)

        self.upconv6 = upconv(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        self.iconv6  = conv(in_channels=512+256,out_channels=512,kernel_size=3,padding=1)

        self.upconv5 = upconv(in_channels=512,out_channels=256,kernel_size=3,padding=1)
        self.iconv5 = conv(in_channels=256+128,out_channels=256,kernel_size=3,padding=1)

        self.upconv4 = upconv(in_channels=256,out_channels=128,kernel_size=3,padding=1)
        self.iconv4 = conv(in_channels=128+64,out_channels=128,kernel_size=3,padding=1)
        self.disp4 = get_disp(in_channels=128,out_channels=2,kernel_size=3,padding=1)

        self.upconv3 = upconv(in_channels=128,out_channels=64,kernel_size=3,padding=1)
        self.iconv3 = conv(in_channels=64+64+2,out_channels=64,kernel_size=3,padding=1)
        self.disp3 = get_disp(in_channels=64,out_channels=2,kernel_size=3,padding=1)

        self.upconv2 = upconv(in_channels=64,out_channels=32,kernel_size=3,padding=1)
        self.iconv2 = conv(in_channels=32+64+2,out_channels=32,kernel_size=3,padding=1)
        self.disp2 = get_disp(in_channels=32,out_channels=2,kernel_size=3,padding=1)

        self.upconv1 = upconv(in_channels=32,out_channels=16,kernel_size=3,padding=1)
        self.iconv1 = conv(in_channels=16+2,out_channels=16,kernel_size=3,padding=1)
        self.disp1 = get_disp(in_channels=16,out_channels=2,kernel_size=3,padding=1)

        self.up_sample = nn.Upsample(scale_factor=2)

    def forward(self,x):

        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        upconv6 = self.upconv6(conv5)
        concat6 = paddle.concat([upconv6,conv4],axis=1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = paddle.concat([upconv5,conv3],axis=1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = paddle.concat([upconv4,conv2],axis=1)
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4(iconv4)
        udisp4 = self.up_sample(disp4)

        upconv3 = self.upconv3(iconv4)
        concat3 = paddle.concat([upconv3,pool1,udisp4],axis=1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3(iconv3)
        udisp3 = self.up_sample(disp3)

        upconv2 = self.upconv2(iconv3)
        concat2 = paddle.concat([upconv2,conv1,udisp3],axis=1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2(iconv2)
        udisp2 = self.up_sample(disp2)

        upconv1 = self.upconv1(iconv2)
        concat1 = paddle.concat([upconv1,udisp2],axis=1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1(iconv1)

        return disp1,disp2,disp3,disp4

class MonodepthModel(nn.Layer):

    def __init__(self,DO_STEREO) -> None:
        super().__init__()

        self.do_stereo = DO_STEREO
        if DO_STEREO:
            self.in_channels = 6
        else:
            self.in_channels = 3

        self.model = ResNet50(in_channels=self.in_channels)

    def process_input(self,left,right):

        if self.do_stereo:
            x = paddle.concat([left,right],axis=1)
        else:
            x = left
        
        return x

    def forward(self,left,right):

        x = self.process_input(left,right)

        x = self.model(x)

        return x

if __name__ == '__main__':

    model = MonodepthModel(DO_STEREO=True)
    l = paddle.rand((1,3,256,512))
    r = paddle.rand((1,3,256,512))
    r = model(l,r)
    print(r[0].shape)
