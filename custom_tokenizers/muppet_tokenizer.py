from typing import List, Optional
from transformers import RobertaTokenizer, AddedToken

class RobertaMuppetTokenizer(RobertaTokenizer):

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        mrc_token="madeupword0000",
        com_token="madeupword0001",
        cls_task_token = "madeupword0002",
        add_prefix_space=False,
        **kwargs):
        mrc_token = AddedToken(mrc_token, lstrip=False, rstrip=False) if isinstance(mrc_token, str) else mrc_token
        com_token = AddedToken(com_token, lstrip=False, rstrip=False) if isinstance(com_token, str) else com_token
        cls_task_token = AddedToken(cls_task_token, lstrip=False, rstrip=False) if isinstance(cls_task_token, str) else cls_task_token
        special_tokens_dict  = { "additional_special_tokens" : [mrc_token, com_token, cls_task_token]}
        
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_task_token=cls_task_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self.add_special_tokens(special_tokens_dict)
        #model.resize_token_embeddings(len(tokenizer))

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if len(token_ids_0) > 0 and token_ids_0[0] in self.additional_special_tokens_ids:
            if token_ids_1 is None:
                return token_ids_0 + [self.sep_token_id]
            return token_ids_0 + sep + sep + token_ids_1 + sep
            
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

def main():
    tokenizer = RobertaMuppetTokenizer.from_pretrained("roberta-base")
    print(tokenizer("madeupword0000Hello_World")["input_ids"])
    print(tokenizer.decode(tokenizer("madeupword0000Hello_World")["input_ids"]))
    
if __name__ == "__main__":
    main()