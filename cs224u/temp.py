from collections import Counter

def BPE(d, N):
    vocab = {chr(i): i for i in range(256)}
    result = [vocab[char] for char in d]
    while len(vocab) < N:
        cnt = Counter()
        for i in range(len(result) - 1):
            cnt[(result[i], result[i + 1],)] += 1
        if len(cnt) <= 0:
            break
        most_common = cnt.most_common(1)[0][0]
        vocab[most_common] = len(vocab)
        new_result = []
        for ent in result:
            if len(new_result) > 0 and new_result[-1] == most_common[0] and ent == most_common[1]:
                new_result[-1] = vocab[most_common]
            else:
                new_result.append(ent)
        result = new_result

    return result, vocab

print(BPE("Large Language Models", 300))