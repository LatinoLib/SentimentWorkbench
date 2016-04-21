using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Latino;
using Latino.Model;
using Latino.Model.Eval;
using Latino.TextMining;
using TextTask.Classifier;

namespace TextTask
{

    public class Tweet
    {
        public ulong Id { get; set; }
        public DateTime Date { get; set; }
        public string Query { get; set; }
        public string User { get; set; }
        public string Text { get; set; }

        public int DisagreementRank { get; set; } // 0 - all agree; 1 - disagree on neutral-pos/neg; 2 - disagree on pos-neg
        public int AgreementRank { get; set; } // number of annotators ageeing on a tweet, 0 - disagreement, 1 - one annotation, 2 - two agree, 3 - ...
    }
    
    public enum SentimentLabel
    {
        Negative = -1,
        Neutral = 0,
        Positive = 1,
        Exclude = 10
    }

    public class SentimentDeltaBowSpace : DeltaBowSpace<SentimentLabel> 
    {
        public SentimentDeltaBowSpace()
        {
        }

        public SentimentDeltaBowSpace(BinarySerializer reader) : base(reader)
        {
        }
    }

    public enum ExecutionMode
    {
        SingleThreaded,
        FoldParallel,
        ModelParallel,
        FoldParallelSmart,
        ModelParallelSmart
    }

    [Flags]
    public enum ModelActions
    {
        Validate = 1 << 0,
        Save = 1 << 1,
        Test = 1 << 2,
        BuildPerf = 1 << 3
    }

    [Flags]
    public enum ProgramModel
    {
        None = 0,
        NaiveBayes = 1 << 0,
        NeutralZone = 1 << 1,
        NeutralZoneAuto = 1 << 2,
        SvmTwoPlane = 1 << 3,
        SvmThreePlaneOneVsOne = 1 << 4,
        SvmThreePlaneOneVsAll = 1 << 5,
        SvmReplication = 1 << 6,
        SvmThreePlaneOneVsOneVoting = 1 << 7,
        NeutralZoneReliability = 1 << 8,
        SvmTwoPlaneBias = 1 << 9,
        TripleSvmVoting = 1 << 10,
        SvmThreePlaneOneVsAllVoting = 1 << 11,
        NeutralZoneBin = 1 << 12,
        DeltaBowSpace = 1 << 13,
        SvmThreePlaneOneVsOneBinVoting = 1 << 14,

        SvmTwoPlaneBW1 = 1 << 15,
        SvmTwoPlaneBW2 = 1 << 16,
        SvmTwoPlaneBW3 = 1 << 17,
        Cascading = 1 << 18
    }


    public static class SentimentUtils
    {
        public static SvmBinaryClassifier<SentimentLabel> CreateSvmModel()
        {
            return new SvmBinaryClassifier<SentimentLabel>
                {
                    BiasedCostFunction = false,  // THIS NEEDS TO BE FALSE, OTHERWISE THE TARGET CLASS DISTRIBUTION IS FCUKED UP
                    BiasedHyperplane = true,
                    C = 1,
                    KernelType = SvmLightKernelType.Linear,
                    VerbosityLevel = SvmLightVerbosityLevel.Off
                };
        }

        public static SvmMulticlassClassifier<SentimentLabel> CreateMulticlassSvmModel()
        {
            return new SvmMulticlassClassifier<SentimentLabel>
                {
                    C = 1
                };
        }

        public static T CreateBowSpace<T>(bool rmvStopWords, int maxNGramLen, WordWeightType wordWeightType, Language language, int minWordFreq) where T : BowSpace, new()
        {
            Set<string>.ReadOnly langStopWords = null;
            IStemmer stemmer = null;
            try
            {
                TextMiningUtils.GetLanguageTools(language, out langStopWords, out stemmer);
            }
            catch
            {
            }

            if (language == Language.Portuguese)
            {
                stemmer = null;
            }

            var bowSpc = new T
                {
                    //Logger = logger,
                    Tokenizer = new RegexTokenizer
                        {
                            TokenRegex =
                                new[] { Language.Russian, Language.Bulgarian, Language.Serbian }.Contains(language)
                                    ? @"[#@$]?([\d_]*[\p{IsBasicLatin}\p{IsLatin-1Supplement}\p{IsLatinExtended-A}\p{IsLatinExtended-B}\p{IsLatinExtendedAdditional}\p{IsCyrillic}\p{IsCyrillicSupplement}-[^\p{L}]][\d_]*){2,}"
                                    : @"[#@$]?([\d_]*[\p{IsBasicLatin}\p{IsLatin-1Supplement}\p{IsLatinExtended-A}\p{IsLatinExtended-B}\p{IsLatinExtendedAdditional}-[^\p{L}]][\d_]*){2,}",
                            IgnoreUnknownTokens = true
                        },
                    CutLowWeightsPerc = 0,
                    MaxNGramLen = maxNGramLen,
                    MinWordFreq = minWordFreq,
                    WordWeightType = wordWeightType,
                    NormalizeVectors = true,
                    Stemmer = stemmer
                };

            if (langStopWords != null)
            {
                var stopWords = new Set<string>(langStopWords) { "rt" };
                // additional stop words
                if (language == Language.English)
                {
                    stopWords.AddRange("im,youre,hes,shes,its,were,theyre,ive,youve,weve,theyve,youd,hed,theyd,youll,theyll,isnt,arent,wasnt,werent,hasnt,havent,hadnt,doesnt,dont,didnt,wont,wouldnt,shant,shouldnt,cant,couldnt,mustnt,lets,thats,whos,whats,heres,theres,whens,wheres,whys,hows,i,m,you,re,he,s,she,it,we,they,ve,d,ll,isn,t,aren,wasn,weren,hasn,haven,hadn,doesn,don,didn,won,wouldn,shan,shouldn,can,couldn,mustn,let,that,who,what,here,there,when,where,why,how".Split(','));
                }
                if (rmvStopWords) { bowSpc.StopWords = stopWords; }
            }
            return bowSpc;
        }

        public static string GetObjectReport(object o, params string[] exclude)
        {
            return o.GetType().Name + "\n" + Utils.GetPropertyValues(o, exclude)
                .Aggregate("", (a, b) => String.Format("{0}{1} = {2}\n", a, b.Key, (b.Value == null ? "null" : b.Value.ToString())));
        }

        public static ProgramModel GetProgramModel(IModel<SentimentLabel, SparseVector<double>> model)
        {
            ProgramModel m;
            if (model is NeutralZoneClassifier)
            {
                m = ProgramModel.NeutralZone;
            }
            else if (model is TwoPlaneClassifier)
            {
                m = ProgramModel.SvmTwoPlane;
            }
            else if (model is ThreePlaneOneVsOneClassifier)
            {
                m = ProgramModel.SvmThreePlaneOneVsOne;
            }
            else if (model is ThreePlaneOneVsAllClassifier)
            {
                m = ProgramModel.SvmThreePlaneOneVsAll;
            }
            else if (model is NeutralZoneReliabilityClassifier)
            {
                m = ProgramModel.NeutralZoneReliability;
            }
            else if (model is ThreePlaneOneVsOneVotingClassifier)
            {
                m = ProgramModel.SvmThreePlaneOneVsOneVoting;
            }
            else if (model is ThreePlaneOneVsAllVotingClassifier)
            {
                m = ProgramModel.SvmThreePlaneOneVsAllVoting;
            }
            else
            {
                m = ProgramModel.None;
            }

            return m;
        }
    }


    public class BowReinitCrossValidator : TaskMappingCrossValidator<SentimentLabel, Tweet, SparseVector<double>>
    {
        private readonly ConcurrentDictionary<int, BowSpace> mFoldBowSpaces = new ConcurrentDictionary<int, BowSpace>();
        private readonly ConcurrentDictionary<Tuple<int, int>, IModel<SentimentLabel, SparseVector<double>>> mFoldModels =
            new ConcurrentDictionary<Tuple<int, int>, IModel<SentimentLabel, SparseVector<double>>>();

        public BowReinitCrossValidator()
        {
        }

        public BowReinitCrossValidator(IEnumerable<Func<IModel<SentimentLabel, SparseVector<double>>>> modelFuncs)
            : base(modelFuncs)
        {
        }

        public Func<BowSpace> BowSpaceFunc { get; set; }
        public Dictionary<Tuple<int, int>, IModel<SentimentLabel, SparseVector<double>>> FoldModels
        {
            get
            {
                return mFoldModels.ToDictionary(kv => kv.Key, kv => kv.Value);
            }
        }

        protected override ILabeledDataset<SentimentLabel, SparseVector<double>> MapTrainSet(int foldN, ILabeledDataset<SentimentLabel, Tweet> trainSet)
        {
            BowSpace bowSpace;
            Preconditions.CheckState(!mFoldBowSpaces.TryGetValue(foldN, out bowSpace));
            Preconditions.CheckState(mFoldBowSpaces.TryAdd(foldN, bowSpace = BowSpaceFunc()));

            List<SparseVector<double>> bowData = bowSpace is DeltaBowSpace<SentimentLabel>
                ? ((DeltaBowSpace<SentimentLabel>)bowSpace).Initialize(new LabeledDataset<SentimentLabel, string>(trainSet
                    .Select(d => new LabeledExample<SentimentLabel, string>(d.Label, d.Example.Text))))
                : bowSpace.Initialize(trainSet.Select(d => d.Example.Text));

            var bowDataset = new LabeledDataset<SentimentLabel, SparseVector<double>>(); 
            for (int i = 0; i < bowData.Count; i++)
            {
                bowDataset.Add(trainSet[i].Label, bowData[i]);
            }

            return bowDataset;
        }

        protected override ILabeledDataset<SentimentLabel, SparseVector<double>> MapTestSet(int foldN, ILabeledDataset<SentimentLabel, Tweet> testSet)
        {
            return new LabeledDataset<SentimentLabel, SparseVector<double>>(testSet.Select(le =>
            {
                SparseVector<double> sparseVector = mFoldBowSpaces[foldN].ProcessDocument(le.Example.Text);
                return new LabeledExample<SentimentLabel, SparseVector<double>>(le.Label, sparseVector);
            }));
        }

        protected override ILabeledDataset<SentimentLabel, SparseVector<double>> BeforeTrain(int foldN, IModel<SentimentLabel, SparseVector<double>> model,
            ILabeledDataset<SentimentLabel, Tweet> trainSet, ILabeledDataset<SentimentLabel, SparseVector<double>> mappedTrainSet)
        {
            mappedTrainSet = base.BeforeTrain(foldN, model, trainSet, mappedTrainSet);

            // replication wrapper needs special treatment
            if (model is ReplicationWrapperClassifier)
            {
                ((ReplicationWrapperClassifier)model).BowSpace = mFoldBowSpaces[foldN];
            }

            // add fold's models for report
            for (int i = 0; !mFoldModels.TryAdd(new Tuple<int, int>(foldN, i), model); i++) { }

            return mappedTrainSet;
        }

        public Dictionary<int, BowSpace> GetBowSpaces()
        {
            return mFoldBowSpaces.ToDictionary(kv => kv.Key, kv => kv.Value);
        }
    }
}