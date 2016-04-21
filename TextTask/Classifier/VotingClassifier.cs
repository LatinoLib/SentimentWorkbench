using System;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;

namespace TextTask.Classifier
{
    public abstract class VotingClassifier<LblT, ExT> : IModel<LblT, ExT>, IDisposable
    {
        private Dictionary<string, VotingEntry> mVotingEntries;
        private IModel<LblT, ExT>[] mInnerModels;

        protected VotingClassifier(IModel<LblT, ExT>[] innerModels)
        {
            Preconditions.CheckArgument(typeof(LblT).IsEnum);
            Preconditions.CheckNotNull(innerModels);
            Preconditions.CheckArgument(innerModels.Length > 0);

            mInnerModels = innerModels;
            mVotingEntries = new Dictionary<string, VotingEntry>();
            foreach (List<LblT> permutation in GetPermutations(innerModels.Length))
            {
                mVotingEntries.Add(StringOf(permutation), new VotingEntry());
            }
        }

        protected VotingClassifier(BinarySerializer reader)
        {
            Load(reader);
        }

        public void Save(BinarySerializer writer)
        {
            mVotingEntries.SaveDictionary(writer);
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
            mVotingEntries = reader.LoadDictionary<string, VotingEntry>();
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
                string key = StringOf(mInnerModels.Select(m => m.Predict(le_.Example).BestClassLabel));
                VotingEntry votingEntry = mVotingEntries[key];
                votingEntry.LabelCounts[le.Label]++;
            }
            foreach (VotingEntry entry in mVotingEntries.Values)
            {
                PerformVoting(entry);
            }

            IsTrained = true;
        }

        public Prediction<LblT> Predict(ExT example)
        {
            Preconditions.CheckState(IsTrained);

            string key = StringOf(mInnerModels.Select(m => m.Predict(example).BestClassLabel));
            VotingEntry entry = mVotingEntries[key];
            return new Prediction<LblT>(new[] { new KeyDat<double, LblT>(1.0, entry.Label) });
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
    
        private static string StringOf(IEnumerable<LblT> labels)
        {
            return string.Join("-", labels);
        }

        private static List<List<LblT>> GetPermutations(int len)
        {
            var result = new List<List<LblT>>();
            foreach (LblT right in Enum.GetValues(typeof(LblT)).Cast<LblT>())
            {
                if (len > 1)
                {
                    foreach (List<LblT> left in GetPermutations(len - 1))
                    {
                        left.Add(right);
                        result.Add(left);
                    }
                }
                else
                {
                    result.Add(new List<LblT> { right });
                }
            }
            return result;
        }

        protected virtual void PerformVoting(VotingEntry votingEntry)
        {
            votingEntry.Label = votingEntry.LabelCounts.OrderByDescending(kv => kv.Value).First().Key;
        }

        protected class VotingEntry : ISerializable
        {
            public VotingEntry(params LblT[] labels)
            {
                Labels = Preconditions.CheckNotNull(labels);
                LabelCounts = new Dictionary<LblT, int>();
                foreach (LblT label in Enum.GetValues(typeof(LblT)).Cast<LblT>())
                {
                    LabelCounts.Add(label, 0);
                }
            }

            public VotingEntry(BinarySerializer reader)
            {
                Load(reader);
            }

            public LblT[] Labels { get; private set; }
            public Dictionary<LblT, int> LabelCounts { get; private set; }
            public LblT Label { get; set; }
            
            public Dictionary<LblT, double> LabelProbs
            {
                get
                {
                    double sum = LabelCounts.Values.Sum();
                    return LabelCounts.ToDictionary(kv => kv.Key, kv => (double)(kv.Value + 1) / (sum + LabelCounts.Count)); // Laplace
                }
            }

            public double Entropy
            {
                get
                {
                    return -LabelProbs.Values.Sum(v => v * Math.Log(v, 2));
                }
            }

            public override string ToString()
            {
                return string.Format("[{0}] = {1} ({2:0.00})", string.Join(", ", LabelProbs.Select(kv => string.Format("{0}: {1:0.00}", kv.Key, kv.Value))), Label, Entropy);
            }

            public void Save(BinarySerializer writer)
            {
                writer.WriteInt(Labels.Length);
                foreach (LblT label in Labels)
                {
                    writer.WriteObject(label);
                }
                writer.WriteObject(Label);
                LabelCounts.SaveDictionary(writer);
            }

            public void Load(BinarySerializer reader)
            {
                Labels = new LblT[reader.ReadInt()];
                for (int i = 0; i < Labels.Length; i++)
                {
                    reader.ReadObject(typeof(LblT));
                }
                Label = reader.ReadObject<LblT>();
                LabelCounts = reader.LoadDictionary<LblT, int>();
            }
        }

        public void Dispose()
        {
            foreach (IDisposable disposable in mInnerModels.Where(c => c is IDisposable).Cast<IDisposable>())
            {
                disposable.Dispose();
            }
        }

        public override string ToString()
        {
            return string.Join("\n", mVotingEntries.Select(kv => string.Format("{0} \t {1}", kv.Key, kv.Value)));
        }
    }


    public class DelegatingVotingClassifier<LblT, ExT> : VotingClassifier<LblT, ExT>
    {
        public DelegatingVotingClassifier(IModel<LblT, ExT>[] innerModels)
            : base(innerModels)
        {
        }

        public DelegatingVotingClassifier(BinarySerializer reader)
            : base(reader)
        {
        }

        public delegate LabeledDataset<LblT, ExT> TrainSetHandle(int modelIdx, IModel<LblT, ExT> model, LabeledDataset<LblT, ExT> trainSet);
        public TrainSetHandle OnGetTrainSet { get; set; }

        protected override LabeledDataset<LblT, ExT> GetTrainSet(int modelIdx, IModel<LblT, ExT> model, LabeledDataset<LblT, ExT> trainSet)
        {
            Preconditions.CheckNotNull(OnGetTrainSet);
            return OnGetTrainSet(modelIdx, model, trainSet);
        }
    }
}