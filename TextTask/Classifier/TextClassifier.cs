using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.TextMining;

namespace TextTask.Classifier
{
    public class TextClassifier<LblT> : IModel<LblT, string>, IDisposable
    {
        public TextClassifier()
        {
        }

        public TextClassifier(BinarySerializer reader)
        {
            Load(reader);
        }

        public string Description { get; set; }
        public TextFeatureProcessor FeatureProcessor { get; set; }
        public BowSpace BowSpace { get; set; }
        public Action<TextClassifier<LblT>, LabeledDataset<LblT, SparseVector<double>>> OnTrainModel { get; set; }

        public IModel<LblT> Model { get; set; }

        public Type RequiredExampleType { get { return typeof(string); } }
        public bool IsTrained { get; private set; }

        public void Train(ILabeledExampleCollection<LblT> dataset)
        {
            Train((ILabeledExampleCollection<LblT, string>)dataset);
        }

        public Prediction<LblT> Predict(object example)
        {
            return Predict((string)example);
        }

        public void Train(ILabeledExampleCollection<LblT, string> dataset)
        {
            Preconditions.CheckState(!IsTrained);
            Preconditions.CheckNotNull(dataset);
            Preconditions.CheckNotNull(BowSpace);
            Preconditions.CheckNotNull(FeatureProcessor);
            Preconditions.CheckNotNull(Model);

            // preprocess the text
            foreach (LabeledExample<LblT, string> le in dataset)
            {
                le.Example = FeatureProcessor.Run(le.Example);
            }

            // bow vectors
            List<SparseVector<double>> bowData = BowSpace is DeltaBowSpace<LblT>
                ? (BowSpace as DeltaBowSpace<LblT>).Initialize(dataset as ILabeledDataset<LblT, string> ?? new LabeledDataset<LblT, string>(dataset))
                : BowSpace.Initialize(dataset.Select(d => d.Example));
            var bowDataset = new LabeledDataset<LblT, SparseVector<double>>();
            for (int i = 0; i < bowData.Count; i++)
            {
                bowDataset.Add(dataset[i].Label, bowData[i]);
            }

            // train
            if (OnTrainModel == null)
            {
                Model.Train(bowDataset);
            }
            else
            {
                OnTrainModel(this, bowDataset);
            }

            IsTrained = true;
        }

        public Prediction<LblT> Predict(string example)
        {
            Preconditions.CheckState(IsTrained);

            example = FeatureProcessor.Run(example);
            SparseVector<double> vector = BowSpace.ProcessDocument(example);
            
            return Model.Predict(vector);
        }

        public void Save(BinarySerializer writer)
        {
            writer.WriteString(Description);
            FeatureProcessor.Save(writer);
            writer.WriteObject(BowSpace);
            writer.WriteObject(Model);
            writer.WriteBool(IsTrained);
        }

        public void Load(BinarySerializer reader)
        {
            Description = reader.ReadString();
            FeatureProcessor = new TextFeatureProcessor(reader);
            BowSpace = reader.ReadObject<BowSpace>();
            System.Diagnostics.Trace.WriteLine("readObject " + typeof(IModel<LblT>) == null ? "null" : typeof(IModel<LblT>).FullName);
            Model = reader.ReadObject<IModel<LblT>>();
            IsTrained = reader.ReadBool();
        }

        public void Dispose()
        {
            if (Model is IDisposable)
            {
                ((IDisposable)Model).Dispose();
            }
        }
    }
}