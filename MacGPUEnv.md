# How to use GPU on Mac M1/2

1. Install xcode tools

```
xcode-select --install
```

2. Install llvm

```
brew install llvm libomp
```

Edit '/etc/profile' or 'zshrc'
```
export CC=/opt/homebrew/opt/llvm/bin/clang 
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib" 
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
```

source '/etc/profile' or 'zshrc'


3. Install PyTorch for M1/2

*P.S. Please uninstall old PyTorch first.*

```
pip uninstall torch torchvision torchaudio
```

```
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```