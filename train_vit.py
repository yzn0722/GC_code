
import timm
if __name__ == '__main__':
    all_pretrained_models_available = timm.list_models(pretrained=True)
    print(all_pretrained_models_available)
    for i in all_pretrained_models_available:
        print(i)
