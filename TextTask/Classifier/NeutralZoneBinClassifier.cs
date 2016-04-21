using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.Model.Eval;

namespace TextTask.Classifier
{

    public class NeutralZoneBinClassifier : BaseClassifier<SentimentLabel>
    {
        private IModel<SentimentLabel, SparseVector<double>> mBinModel;

        public NeutralZoneBinClassifier()
        {
            BinWidth = 0.01;
        }

        public NeutralZoneBinClassifier(BinarySerializer reader)
        {
            Load(reader);
        }

        public double BinWidth { get; set; }
        public TagDistrTable<SentimentLabel> TagDistrTable { get; private set; }

        public override void Train(ILabeledExampleCollection<SentimentLabel, SparseVector<double>> dataset)
        {
            Preconditions.CheckNotNull(dataset);
            Preconditions.CheckArgumentRange(TagDistrTable == null || TagDistrTable.NumOfDimensions == 2);

            mBinModel = CreateModel();
            mBinModel.Train(new LabeledDataset<SentimentLabel, SparseVector<double>>(dataset.Where(le => le.Label != SentimentLabel.Neutral)));

            TagDistrTable = new EnumTagDistrTable<SentimentLabel>(1, BinWidth, -5, 5, SentimentLabel.Exclude)
                {
                    CalcDistrFunc = (tagCounts, values, tag) => ((double)tagCounts[tag] + 1) / (tagCounts.Values.Sum() + tagCounts.Count) // use Laplace formula
                };
            foreach (LabeledExample<SentimentLabel, SparseVector<double>> le in dataset)
            {
                Prediction<SentimentLabel> prediction = mBinModel.Predict(le.Example);
                TagDistrTable.AddCount(le.Label, prediction.BestClassLabel == SentimentLabel.Positive ? prediction.BestScore : -prediction.BestScore);
            }
            TagDistrTable.Calculate();

            IsTrained = true;
        }

        public override Prediction<SentimentLabel> Predict(SparseVector<double> example)
        {
            Preconditions.CheckState(IsTrained);

            Prediction<SentimentLabel> prediction = mBinModel.Predict(example);

            Dictionary<SentimentLabel, double?> distrValues = TagDistrTable.GetDistrValues(
                prediction.BestClassLabel == SentimentLabel.Positive ? prediction.BestScore : -prediction.BestScore);
            return new Prediction<SentimentLabel>(distrValues
                .OrderByDescending(kv => kv.Value)
                .Select(kv => new KeyDat<double, SentimentLabel>(kv.Value.GetValueOrDefault(), kv.Key)));
        }

        public override void Save(BinarySerializer writer)
        {
            Preconditions.CheckNotNull(TagDistrTable);

            writer.WriteDouble(BinWidth);
            TagDistrTable.Save(writer);
            writer.WriteObject(mBinModel);
            writer.WriteBool(IsTrained);
        }

        public override sealed void Load(BinarySerializer reader)
        {
            BinWidth = reader.ReadDouble();
            TagDistrTable = new TagDistrTable<SentimentLabel>(reader);
            mBinModel = reader.ReadObject<IModel<SentimentLabel, SparseVector<double>>>();
            IsTrained = reader.ReadBool();
        }

        protected override IEnumerable<IDisposable> GetDisposables()
        {
            return new[] { mBinModel }.Where(c => c is IDisposable).Cast<IDisposable>();
        }
    }
}