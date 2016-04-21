using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public class CascadingClassifier<LblT, ExT> : IModel<LblT, ExT>
    {
        public CascadingClassifier(IModel<LblT, ExT> model, LblT label, IModel<LblT, ExT> model1) 
            : this(new[]
                {
                    new ModelLabel { Model = model, Label = label }, 
                    new ModelLabel { Model = model1 }
                })
        {
        }

        public CascadingClassifier(IModel<LblT, ExT> model, LblT label, IModel<LblT, ExT> model1, LblT label1, IModel<LblT, ExT> model2) 
            : this(new[]
                {
                    new ModelLabel { Model = model, Label = label }, 
                    new ModelLabel { Model = model1, Label = label1 },
                    new ModelLabel { Model = model2 }
                })
        {
        }

        public CascadingClassifier(IEnumerable<ModelLabel> modelLabels)
        {
            ModelLabels = Preconditions.CheckNotNull(modelLabels).ToArray();
        }

        public ModelLabel[] ModelLabels { get; private set; }

        public Type RequiredExampleType { get { return typeof(ExT); } }
        public bool IsTrained { get; private set; }

        public void Save(BinarySerializer writer)
        {
            throw new NotImplementedException();
        }

        public void Train(ILabeledExampleCollection<LblT> dataset)
        {
            Train((ILabeledExampleCollection<LblT, ExT>)dataset);
        }

        public Prediction<LblT> Predict(object example)
        {
            return Predict((ExT)example);
        }

        public void Train(ILabeledExampleCollection<LblT, ExT> dataset)
        {
            foreach (ModelLabel modelLabel in ModelLabels.Take(ModelLabels.Count() - 1))
            {
                modelLabel.Model.Train(dataset);
                ModelLabel modelLabel_ = modelLabel;
                dataset = new LabeledDataset<LblT, ExT>(dataset.Where(le => !le.Label.Equals(modelLabel_.Label)));
            }
            ModelLabels.Last().Model.Train(dataset);
            IsTrained = true;
        }

        public Prediction<LblT> Predict(ExT example)
        {
            foreach (ModelLabel modelLabel in ModelLabels.Take(ModelLabels.Count() - 1))
            {
                Prediction<LblT> prediction = modelLabel.Model.Predict(example);
                if (prediction.BestClassLabel.Equals(modelLabel.Label))
                {
                    return prediction;
                }
            }
            return ModelLabels.Last().Model.Predict(example);
        }

        public class ModelLabel
        {
            public IModel<LblT, ExT> Model { get; set; }
            public LblT Label { get; set; }
        }
    }
}