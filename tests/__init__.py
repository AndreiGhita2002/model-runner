# tests package
import warnings

# Suppress PyTorch internal FutureWarning about LeafSpec deprecation
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*", category=FutureWarning)
