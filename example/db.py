import numpy as np
import pickle as cPickle

def add_embeddings(embedding, label,
                   embeddings_path="face_embeddings.npy",
                   labels_path="labels.pickle"):
    first_time = False
    try:
        embeddings = np.load(embeddings_path)
        with open(labels_path,'rb') as f:
            labels = cPickle.load(f)
#    except IOError:
#        raise 
#    except OSError:
#        raise
#    except EOFError:
#        raise
    except Exception as e:
        print (e)
        first_time = True

    if first_time:
        embeddings = embedding
        labels = [label]
    else:
        embeddings = np.concatenate([embeddings, embedding], axis=0)
        labels.append(label)

    np.save(embeddings_path, embeddings)
    with open(labels_path, "wb") as f:
        cPickle.dump(labels, f)

    return True