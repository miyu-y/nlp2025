import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
import os
import torch.nn.functional as F
from transformers import AutoModel


class NoDocModel(nn.Module):

    def __init__(self, base_model, question_encoder=None, generator=None):
        super(NoDocModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size, 2)
        self.question_encoder = question_encoder
        self.generator = generator

    def forward(self, input_ids, attention_mask=None, labels=None):
        text_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)[1]  # pooler_output
        text_output = self.dropout(text_output)

        logits = self.classifier(text_output)

        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return ModelOutput(logits=logits, loss=loss, output=text_output)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        if self.question_encoder is not None:
            self.question_encoder.save_pretrained(os.path.join(save_directory, "question_encoder"))
        if self.generator is not None:
            self.generator.save_pretrained(os.path.join(save_directory, "generator"))

        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, base_model, save_directory):
        question_encoder = None
        generator = None

        if os.path.exists(os.path.join(save_directory, "question_encoder")):
            question_encoder = AutoModel.from_pretrained(os.path.join(save_directory, "question_encoder"))
        if os.path.exists(os.path.join(save_directory, "generator")):
            generator = AutoModel.from_pretrained(os.path.join(save_directory, "generator"))

        model = cls(base_model=base_model, question_encoder=question_encoder, generator=generator)
        model.load_state_dict(torch.load(os.path.join(save_directory, "pytorch_model.bin")))

        return model


class WithDocModel(nn.Module):
    def __init__(self, base_model, question_encoder=None, generator=None):
        super(WithDocModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size * 2, 2)
        self.question_encoder = question_encoder
        self.generator = generator

    def forward(self, input_ids, attention_mask=None, labels=None):

        ref_input_ids = input_ids[0]
        text_input_ids = input_ids[1]
        ref_attention_mask = attention_mask[0]
        text_attention_mask = attention_mask[1]

        ref_output = self.base_model(input_ids=ref_input_ids, attention_mask=ref_attention_mask)[1]
        text_output = self.base_model(input_ids=text_input_ids, attention_mask=text_attention_mask)[1]

        ref_output = self.dropout(ref_output)
        text_output = self.dropout(text_output)

        logits = self.classifier(torch.cat([ref_output, text_output], dim=1))

        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None

        return ModelOutput(logits=logits, loss=loss, output=[ref_output, text_output])

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        if self.question_encoder is not None:
            self.question_encoder.save_pretrained(os.path.join(save_directory, "question_encoder"))
        if self.generator is not None:
            self.generator.save_pretrained(os.path.join(save_directory, "generator"))

        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, base_model, save_directory):
        question_encoder = None
        generator = None

        if os.path.exists(os.path.join(save_directory, "question_encoder")):
            question_encoder = AutoModel.from_pretrained(os.path.join(save_directory, "question_encoder"))
        if os.path.exists(os.path.join(save_directory, "generator")):
            generator = AutoModel.from_pretrained(os.path.join(save_directory, "generator"))

        model = cls(base_model=base_model, question_encoder=question_encoder, generator=generator)
        model.load_state_dict(torch.load(os.path.join(save_directory, "pytorch_model.bin")))

        return model


class TripletModel(nn.Module):
    def __init__(self, base_model, loss_function, question_encoder=None, generator=None):
        super(TripletModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size * 2, 2)
        self.loss_function = loss_function
        self.question_encoder = question_encoder
        self.generator = generator

    def forward(self, input_ids, attention_mask=None, labels=None):

        anchor_input_ids = input_ids[0]
        positive_input_ids = input_ids[1]
        negative_input_ids = input_ids[2]
        anchor_attention_mask = attention_mask[0]
        positive_attention_mask = attention_mask[1]
        negative_attention_mask = attention_mask[2]

        anchor_output = self.base_model(
            input_ids=anchor_input_ids, attention_mask=anchor_attention_mask, return_dict=True
        )[1]
        positive_output = self.base_model(
            input_ids=positive_input_ids, attention_mask=positive_attention_mask, return_dict=True
        )[1]
        negative_output = self.base_model(
            input_ids=negative_input_ids, attention_mask=negative_attention_mask, return_dict=True
        )[1]

        anchor_output = self.dropout(anchor_output)
        positive_output = self.dropout(positive_output)
        negative_output = self.dropout(negative_output)

        positive_logits = self.classifier(torch.cat([anchor_output, positive_output], dim=1))
        negative_logits = self.classifier(torch.cat([anchor_output, negative_output], dim=1))

        classification_loss, triplet_loss = self.loss_function(
            anchor_output, positive_output, negative_output, positive_logits, negative_logits
        )
        loss = classification_loss + triplet_loss
        
        if labels is None:
            labels = torch.zeros(len(positive_output), dtype=torch.long, device=positive_output.device)
        outputs = torch.stack(
            [positive_output[i] if labels[i] == 0 else negative_output[i] for i in range(len(labels))]
        )


        return ModelOutput(logits=[positive_logits, negative_logits], loss=loss, output=[anchor_output, outputs])

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        if self.question_encoder is not None:
            self.question_encoder.save_pretrained(os.path.join(save_directory, "question_encoder"))
        if self.generator is not None:
            self.generator.save_pretrained(os.path.join(save_directory, "generator"))

        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, base_model, loss_function, save_directory):
        question_encoder = None
        generator = None

        if os.path.exists(os.path.join(save_directory, "question_encoder")):
            question_encoder = AutoModel.from_pretrained(os.path.join(save_directory, "question_encoder"))
        if os.path.exists(os.path.join(save_directory, "generator")):
            generator = AutoModel.from_pretrained(os.path.join(save_directory, "generator"))

        model = cls(
            base_model=base_model, loss_function=loss_function, question_encoder=question_encoder, generator=generator
        )
        model.load_state_dict(torch.load(os.path.join(save_directory, "pytorch_model.bin")))

        return model
