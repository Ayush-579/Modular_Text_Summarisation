import logging
from summarisation.preprocessing import preprocess_dialogue, preprocess_summary
from summarisation.postprocessing import postprocess_summary

class SummarizationPipeline:
    def __init__(self, loader, model):
        self.loader = loader
        self.model = model
    
    def run(self, num_samples=5):
        logging.info("Running summarization pipeline...")
        data = self.loader.load()
        raw_dialogues = [x["dialogue"] for x in data.select(range(num_samples))]
        references = [x["summary"] for x in data.select(range(num_samples))]

    #  Preprocess the dialogues
        inputs = [preprocess_dialogue(d) for d in raw_dialogues]

    #  Generate summaries
        raw_outputs = self.model.summarize(inputs)

    #  Postprocess summaries (optional: also refs if needed for eval)
        outputs = [postprocess_summary(o) for o in raw_outputs]
        return list(zip(raw_dialogues, outputs, references))

