
def get_source():
    return FashionMNIST( 
        root='./datasets/FashionMNIST/',
        train=True,
        download=False,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    )

def get_test():
    return FashionMNIST( 
        root='./datasets/FashionMNIST/',
        train=False,
        download=False,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
    )
