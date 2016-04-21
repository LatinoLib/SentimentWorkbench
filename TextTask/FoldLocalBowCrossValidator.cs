using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.Model.Eval;
using Latino.TextMining;

namespace TextTask
{
    public class FoldLocalBowCrossValidator<LblT> : TaskMappingCrossValidator<LblT, string, SparseVector<double>>
    {
        private readonly ConcurrentDictionary<int, BowSpace> mFoldBowSpaces = new ConcurrentDictionary<int, BowSpace>();
        private readonly ConcurrentDictionary<Tuple<int, int>, IModel<LblT, SparseVector<double>>> mFoldModels =
            new ConcurrentDictionary<Tuple<int, int>, IModel<LblT, SparseVector<double>>>();

        public FoldLocalBowCrossValidator()
        {
        }

        public FoldLocalBowCrossValidator(IEnumerable<Func<IModel<LblT, SparseVector<double>>>> modelFuncs)
            : base(modelFuncs)
        {
        }

        public Func<BowSpace> BowSpaceFunc { get; set; }

        public Dictionary<Tuple<int, int>, IModel<LblT, SparseVector<double>>> FoldModels
        {
            get
            {
                return mFoldModels.ToDictionary(kv => kv.Key, kv => kv.Value);
            }
        }

        public Dictionary<int, BowSpace> FoldBowSpaces
        {
            get
            {
                return mFoldBowSpaces.ToDictionary(kv => kv.Key, kv => kv.Value);
            }
        }

        protected override ILabeledDataset<LblT, SparseVector<double>> MapTrainSet(int foldN, ILabeledDataset<LblT, string> trainSet)
        {
            BowSpace bowSpace;
            Preconditions.CheckState(!mFoldBowSpaces.TryGetValue(foldN, out bowSpace));
            Preconditions.CheckState(mFoldBowSpaces.TryAdd(foldN, bowSpace = BowSpaceFunc()));

            List<SparseVector<double>> bowData = bowSpace is DeltaBowSpace<LblT>
                ? ((DeltaBowSpace<LblT>)bowSpace).Initialize(trainSet)
                : bowSpace.Initialize(trainSet.Select(d => d.Example));

            var bowDataset = new LabeledDataset<LblT, SparseVector<double>>();
            for (int i = 0; i < bowData.Count; i++)
            {
                bowDataset.Add(trainSet[i].Label, bowData[i]);
            }

            return bowDataset;
        }

        protected override ILabeledDataset<LblT, SparseVector<double>> MapTestSet(int foldN, ILabeledDataset<LblT, string> testSet)
        {
            return new LabeledDataset<LblT, SparseVector<double>>(testSet.Select(le =>
            {
                SparseVector<double> sparseVector = mFoldBowSpaces[foldN].ProcessDocument(le.Example);
                return new LabeledExample<LblT, SparseVector<double>>(le.Label, sparseVector);
            }));
        }

        protected override ILabeledDataset<LblT, SparseVector<double>> BeforeTrain(int foldN, IModel<LblT, SparseVector<double>> model, 
            ILabeledDataset<LblT, string> trainSet, ILabeledDataset<LblT, SparseVector<double>> mappedTrainSet)
        {
            mappedTrainSet = base.BeforeTrain(foldN, model, trainSet, mappedTrainSet);

            // add fold's models for report
            for (int i = 0; !mFoldModels.TryAdd(new Tuple<int, int>(foldN, i), model); i++) { }

            return mappedTrainSet;
        }
    }
}