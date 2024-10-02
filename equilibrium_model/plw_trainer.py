from trl import SFTTrainer
import torch

class PLWTrainer(SFTTrainer):

    def __init__(self, *args, prompt_loss_weight=1.0, sep=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_loss_weight = prompt_loss_weight
        self.sep = sep

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        bs = input_ids.shape[0]
        max_len = input_ids.shape[1]
        masks = []
        for i in range(bs):
            ids = input_ids[i]
            index = torch.nonzero(torch.eq(ids, self.sep)).view(-1).max()
            mask = torch.cat((torch.ones(index + 1) * self.prompt_loss_weight, torch.ones(max_len - index - 1)))
            masks.append(mask)
        masks = torch.stack(masks, dim=0)

        outputs = model(**inputs, prompt_loss_mask=masks)

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
