
try:
    from deepface import DeepFace
    print("Import Successful!")
    print("DeepFace version:", DeepFace.__version__)
except ImportError as e:
    print(f"Import Failed: {e}")
    import deepface
    print("deepface package dir:", deepface.__file__)
    print("deepface contents:", dir(deepface))
except Exception as e:
    print(f"Other Error: {e}")
