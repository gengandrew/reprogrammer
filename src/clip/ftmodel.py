import clip
import torch


class ImageEncoder(torch.nn.Module):
    def __init__(self, model, keep_lang=False, mr_resolution=None, up_resolution=None, dropout=None):
        super().__init__()

        if mr_resolution is None or up_resolution is None:
            print('Currently using a normal CLIP model')
            self.model, self.preprocess = clip.load(model)
        else:
            print('Currently using a reprogramming CLIP model')
            self.model, self.preprocess = clip.load('ViT-B/32', mr_resolution=mr_resolution, up_resolution=up_resolution, dropout=dropout)

        # Issue -> https://github.com/openai/CLIP/issues/40
        self.model = self.model.float()

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, weights, biases=None, normalize=True):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        
        return super().forward(inputs)


class LogisticRegressionHead(torch.nn.Module):
    def __init__(self, input_size, output_size, normalize=True):
        super().__init__()

        self.normalize = normalize
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        
        outputs = self.linear(inputs)
        return outputs


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.classification_head = classification_head

        if self.image_encoder is not None:
            self.preprocess = self.image_encoder.preprocess

    def forward(self, inputs):
        inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)

        return outputs


def get_classification_head(clip_model, classnames, labels):
    logit_scale = clip_model.logit_scale
    clip_model.eval()
    clip_model.cuda()

    # Loading the clip embedded textual representations of each class 
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = []
            for t in labels:
                texts.append(t(classname))
            
            texts = clip.tokenize(texts).cuda() # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights = logit_scale.exp() * zeroshot_weights
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    # Defining classification head
    classification_head = ClassificationHead(weights=zeroshot_weights, normalize=True)

    return classification_head