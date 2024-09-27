# How to use GPU on Mac M1/2

1. Install xcode tools

```
xcode-select --install
```

2. Install llvm

```
brew install llvm libomp
```

3. Install PyTorch for M1/2

*P.S. Please uninstall old PyTorch first.*

```
pip uninstall torch torchvision torchaudio
```

```
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```