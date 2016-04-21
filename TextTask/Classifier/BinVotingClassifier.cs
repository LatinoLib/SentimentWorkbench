using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public abstract class BinVotingClassifier<LblT, ExT> : IModel<LblT, ExT>, IDisposable
    {
        private TagDistrTable<LblT> mTagDistrTable;
        private IModel<LblT, ExT>[] mInnerModels;

        protected BinVotingClassifier(IModel<LblT, ExT>[] innerModels, double binWidth = 0.05, params LblT[] excludedTags)
        {
            Preconditions.CheckArgument(typeof(LblT).IsEnum);
            Preconditions.CheckNotNull(innerModels);
            Preconditions.CheckArgument(innerModels.Length > 0);
            Preconditions.CheckArgumentRange(binWidth > 0);

            mInnerModels = innerModels;
            mTagDistrTable = new EnumTagDistrTable<LblT>(innerModels.Length, binWidth, -5, 5, excludedTags)
                {
                    CalcDistrFunc = (tagCounts, values, tag) => CalcLabelScore(tagCounts, values, tag)
                };
        }

        protected BinVotingClassifier(BinarySerializer reader)
        {
            Load(reader);
        }

        public void Save(BinarySerializer writer)
        {
            mTagDistrTable.Save(writer);
            writer.WriteInt(mInnerModels.Length);
            foreach (IModel<LblT, ExT> model in mInnerModels)
            {
                writer.WriteObject(model);
            }
            writer.WriteBool(IsTrained);
        }

        public Type RequiredExampleType { get { return typeof(ExT); } }

        public void Load(BinarySerializer reader)
        {
            mTagDistrTable = new TagDistrTable<LblT>(reader);
            mInnerModels = new IModel<LblT, ExT>[reader.ReadInt()];
            for (int i = 0; i < mInnerModels.Length; i++)
            {
                mInnerModels[i] = reader.ReadObject<IModel<LblT, ExT>>();
            }
            IsTrained = reader.ReadBool();
        }

        public bool IsTrained { get; private set; }

        public void Train(ILabeledExampleCollection<LblT> dataset)
        {
            Train((ILabeledExampleCollection<LblT, ExT>)dataset);
        }

        public Prediction<LblT> Predict(object example)
        {
            return Predict((ExT)example);
        }

        public delegate IModel<LblT, ExT> CreateModelHandler(int modelIdx);

        public CreateModelHandler OnCreateModel { get; set; }

        public void Train(ILabeledExampleCollection<LblT, ExT> dataset)
        {
            Preconditions.CheckNotNull(dataset);

            var trainDataset = new LabeledDataset<LblT, ExT>(dataset);
            for (int i = 0; i < mInnerModels.Length; i++)
            {
                if (mInnerModels[i] == null) { mInnerModels[i] = CreateModel(i); }
                mInnerModels[i].Train(GetTrainSet(i, mInnerModels[i], trainDataset));
            }

            foreach (LabeledExample<LblT, ExT> le in trainDataset)
            {
                LabeledExample<LblT, ExT> le_ = le;
                double[] scores = GetPredictionScores(mInnerModels.Select(m => m.Predict(le_.Example)).ToArray()).ToArray();
                mTagDistrTable.AddCount(le.Label, scores);
            }
            mTagDistrTable.Calculate();

            IsTrained = true;
        }

        public Prediction<LblT> Predict(ExT example)
        {
            Preconditions.CheckState(IsTrained);
            double[] scores = GetPredictionScores(mInnerModels.Select(m => m.Predict(example)).ToArray()).ToArray();
            return GetPrediction(scores, mTagDistrTable);
        }

        protected abstract IEnumerable<double> GetPredictionScores(Prediction<LblT>[] predictions);

        protected virtual Prediction<LblT> GetPrediction(double[] scores, TagDistrTable<LblT> tagDistrTable)
        {
            IEnumerable<KeyDat<double, LblT>> labelProbs = mTagDistrTable.GetDistrValues(scores)
                .OrderByDescending(kv => Math.Abs(kv.Value.GetValueOrDefault()))
                .Select(kv => new KeyDat<double, LblT>(kv.Value.GetValueOrDefault(), kv.Key));

            return new Prediction<LblT>(labelProbs);
        }

        protected virtual double CalcLabelScore(Dictionary<LblT, int> labelCounts, double[] modelScores, LblT label)
        {
            return ((double)labelCounts[label] + 1) / (labelCounts.Values.Sum() + labelCounts.Count); // Laplace
        }

        protected virtual IModel<LblT, ExT> CreateModel(int modelIdx)
        {
            if (OnCreateModel != null)
            {
                return OnCreateModel(modelIdx);
            }
            throw new NotImplementedException();
        }

        protected abstract LabeledDataset<LblT, ExT> GetTrainSet(int modelIdx, IModel<LblT, ExT> model, LabeledDataset<LblT, ExT> trainSet);
    
        public void Dispose()
        {
            foreach (IDisposable disposable in mInnerModels.Where(c => c is IDisposable).Cast<IDisposable>())
            {
                disposable.Dispose();
            }
        }
    }
}