using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Latino;

namespace TextTask.Classifier
{
    public class BinTable<BinT> : ISerializable
    {
        private int mDimensionBinCount;
        private int mBinCount;
        private int[] mDimensionSteps;
        private SparseVector<BinT> mBins;

        public BinTable(int numOfDimensions, double binWidth, double minValue, double maxValue)
        {
            Preconditions.CheckArgumentRange(numOfDimensions > 0);
            Preconditions.CheckArgumentRange(minValue < maxValue);
            Preconditions.CheckArgumentRange(binWidth > 0 && binWidth <= maxValue - minValue);

            BinWidth = binWidth;
            MinValue = minValue;
            MaxValue = maxValue;
            NumOfDimensions = numOfDimensions;

            mDimensionBinCount = (int)Math.Ceiling((MaxValue - MinValue) / BinWidth);

            mBins = new SparseVector<BinT>();

            int dataSize = mBinCount = (int)Math.Pow(mDimensionBinCount, numOfDimensions);
            mDimensionSteps = new int[numOfDimensions];
            for (int i = 0; i < numOfDimensions; i++)
            {
                dataSize /= mDimensionBinCount;
                mDimensionSteps[i] = dataSize;
            }
        }

        public BinTable(BinarySerializer reader)
        {
            Load(reader);
        }

        public double BinWidth { get; private set; }
        public double MinValue { get; private set; }
        public double MaxValue { get; private set; }
        public int NumOfDimensions { get; private set; }

        public BinT this[params double[] values]
        {
            get
            {
                int index = GetIndex(values);
                return mBins.TryGet(index, default(BinT));
            }
            set
            {
                int index = GetIndex(values);
                mBins[index] = value;
            }
        }

        public SparseVector<BinT> Vector()
        {
            return mBins;
        } 

        public int GetIndex(params double[] values)
        {
            Preconditions.CheckNotNull(values);
            Preconditions.CheckArgumentRange(values.Length == NumOfDimensions);

            return mDimensionSteps.Select((ds, i) =>
            {
                int index;
                if (values[i] <= MinValue)
                {
                    index = 0;
                }
                else if (values[i] >= MaxValue)
                {
                    index = mDimensionBinCount - 1;
                }
                else
                {
                    index = (int)Math.Truncate((values[i] - MinValue) / BinWidth);
                }

                int result = ds * index;
                if (result < 0)
                {
                    Console.WriteLine(result);
                    return 0;
                }
                return result;
            }).Sum();
        }

        public double[] GetValues(int index)
        {
            Preconditions.CheckArgumentRange(index >= 0 && index < mBinCount);

            var values = new double[NumOfDimensions];
            for (int i = 0; i < mDimensionSteps.Length; i++)
            {
                values[i] = MinValue + BinWidth * index / mDimensionSteps[i] + BinWidth / 2;
                index %= mDimensionSteps[i];
            }
            return values;
        }

        public void Save(BinarySerializer writer)
        {
            writer.WriteDouble(BinWidth);
            writer.WriteDouble(MinValue);
            writer.WriteDouble(MaxValue);
            writer.WriteInt(NumOfDimensions);
            writer.WriteInt(mDimensionBinCount);
            writer.WriteInt(mBinCount);
            writer.WriteInt(mDimensionSteps.Length);
            foreach (int step in mDimensionSteps)
            {
                writer.WriteInt(step);
            }
            mBins.Save(writer);
        }

        public void Load(BinarySerializer reader)
        {
            BinWidth = reader.ReadDouble();
            MinValue = reader.ReadDouble();
            MaxValue = reader.ReadDouble();
            NumOfDimensions = reader.ReadInt();
            mDimensionBinCount = reader.ReadInt();
            mBinCount = reader.ReadInt();
            mDimensionSteps = new int[reader.ReadInt()];
            for (int i = 0; i < mDimensionSteps.Length; i++)
            {
                mDimensionSteps[i] = reader.ReadInt();
            }
            mBins = new SparseVector<BinT>(reader);
        }
    }


    public class TagDistrTable<TagT> : IEnumerable<int>, ISerializable
    {
        private BinTable<int>[] mTagCounts;
        private BinTable<double>[] mTagDistrs;
        private Dictionary<TagT, int> mTagIndexes;

        public TagDistrTable(int numOfDimensions, double binWidth, double minValue, double maxValue, TagT[] tags)
        {
            Preconditions.CheckNotNull(tags);
            Preconditions.CheckArgumentRange(tags.Length > 1);
            NumOfDimensions = numOfDimensions;
            BinWidth = binWidth;
            MinValue = minValue;
            MaxValue = maxValue;
            Tags = tags;

            mTagCounts = new BinTable<int>[Tags.Length];
            mTagDistrs = new BinTable<double>[Tags.Length];
            mTagIndexes = Tags.Select((t, i) => new { t, i }).ToDictionary(ti => ti.t, ti => ti.i);

            for (int i = 0; i < mTagCounts.Length; i++)
            {
                mTagCounts[i] = new BinTable<int>(NumOfDimensions, BinWidth, MinValue, MaxValue);
                mTagDistrs[i] = new BinTable<double>(NumOfDimensions, BinWidth, MinValue, MaxValue);
            }
        }

        public TagDistrTable(BinarySerializer reader)
        {
            Load(reader);
        }

        public int NumOfDimensions { get; private set; }
        public double BinWidth { get; private set; }
        public double MinValue { get; private set; }
        public double MaxValue { get; private set; }
        public TagT[] Tags { get; private set; }
        
        public Func<Dictionary<TagT, int>, double[], TagT, double> CalcDistrFunc { get; set; }

        public void AddCount(TagT tag, params double[] values)
        {
            AddCount(tag, values, 1);
        }

        public void AddCount(TagT tag, double[] values, int count)
        {
            int tagIndex;
            Preconditions.CheckArgument(mTagIndexes.TryGetValue(tag, out tagIndex));
            mTagCounts[tagIndex][values] += count;
        }

        public void Calculate()
        {
            Preconditions.CheckNotNull(CalcDistrFunc);

            foreach (int index in this)
            {
                Dictionary<TagT, int> tagCounts = mTagIndexes.ToDictionary(kv => kv.Key, kv => mTagCounts[kv.Value].Vector().TryGet(index, 0));
                foreach (KeyValuePair<TagT, int> kv in mTagIndexes)
                {
                    mTagDistrs[kv.Value].Vector()[index] = CalcDistrFunc(tagCounts, mTagCounts[kv.Value].GetValues(index), kv.Key);
                }
            }
        }

        public double? GetDistrValue(TagT tag, params double[] values)
        {
            int tagIndex;
            Preconditions.CheckArgument(mTagIndexes.TryGetValue(tag, out tagIndex));
            int index = mTagDistrs[tagIndex].Vector().GetDirectIdx(mTagDistrs[tagIndex].GetIndex(values));
            return index < 0 ? (double?)null : mTagDistrs[tagIndex].Vector().GetDatDirect(index);
        }

        public void SetDistrValue(TagT tag, double distrValue, params double[] values)
        {
            int tagIndex;
            Preconditions.CheckArgument(mTagIndexes.TryGetValue(tag, out tagIndex));
            mTagDistrs[tagIndex][values] = distrValue;
        }

        public Dictionary<TagT, double?> GetDistrValues(params double[] values)
        {
            return mTagIndexes.ToDictionary(kv => kv.Key, kv =>
            {
                int index = mTagDistrs[kv.Value].Vector().GetDirectIdx(mTagDistrs[kv.Value].GetIndex(values));
                return index < 0 ? (double?)null : mTagDistrs[kv.Value].Vector().GetDatDirect(index);
            });
        }

        public int? GetCount(TagT tag, params double[] values)
        {
            int tagIndex;
            Preconditions.CheckArgument(mTagIndexes.TryGetValue(tag, out tagIndex));
            int index = mTagCounts[tagIndex].Vector().GetDirectIdx(mTagDistrs[tagIndex].GetIndex(values));
            return index < 0 ? (int?)null : mTagCounts[tagIndex].Vector().GetDatDirect(index);
        }

        public Dictionary<TagT, int?> GetCounts(params double[] values)
        {
            return mTagIndexes.ToDictionary(kv => kv.Key, kv =>
            {
                int index = mTagCounts[kv.Value].Vector().GetDirectIdx(mTagDistrs[kv.Value].GetIndex(values));
                return index < 0 ? (int?)null : mTagCounts[kv.Value].Vector().GetDatDirect(index);
            });
        }

        public IEnumerator<int> GetEnumerator()
        {
            IEnumerator<IdxDat<int>>[] enums = mTagCounts.Select(tc => tc.Vector().GetEnumerator()).ToArray();
            IdxDat<int>?[] cursor = enums.Select(e => e.MoveNext() ? e.Current : (IdxDat<int>?)null).ToArray();

            while (cursor.Any(id => id != null))
            {
                int index = cursor.Where(id => id != null).Min(id => id.Value.Idx);
                yield return index;
                for (int i = 0; i < cursor.Length; i++)
                {
                    if (cursor[i] != null && cursor[i].Value.Idx == index)
                    {
                        cursor[i] = enums[i].MoveNext() ? enums[i].Current : (IdxDat<int>?)null;
                    }
                }
            }
        }

        public override string ToString()
        {
            if (mTagCounts == null || mTagCounts.Length == 0) { return ""; }

            var sb = new StringBuilder();
            foreach (int index in this)
            {
                sb.Append(string.Join("\t", mTagCounts[0].GetValues(index))).Append("\t");
                foreach (KeyValuePair<TagT, int> kv in mTagIndexes)
                {
                    sb.Append(kv.Key).Append("\t");
                    sb.Append(mTagCounts[kv.Value].Vector().TryGet(index, 0)).Append("\t");
                    sb.Append(Math.Round(mTagDistrs[kv.Value].Vector().TryGet(index, 0), 3)).Append("\t");
                }
                sb.AppendLine();
            }
            return sb.ToString();
        }

        public void Save(BinarySerializer writer)
        {
            writer.WriteInt(NumOfDimensions);
            writer.WriteDouble(BinWidth);
            writer.WriteDouble(MinValue);
            writer.WriteDouble(MaxValue);
            writer.WriteInt(Tags.Length);
            foreach (TagT tag in Tags)
            {
                writer.WriteObject(tag);
            }
            writer.WriteInt(mTagCounts.Length);
            foreach (BinTable<int> tagCount in mTagCounts)
            {
                tagCount.Save(writer);
            }
            writer.WriteInt(mTagDistrs.Length);
            foreach (BinTable<double> tagDistr in mTagDistrs)
            {
                tagDistr.Save(writer);
            }
            mTagIndexes.SaveDictionary(writer);
        }

        public void Load(BinarySerializer reader)
        {
            NumOfDimensions = reader.ReadInt();
            BinWidth = reader.ReadDouble();
            MinValue = reader.ReadDouble();
            MaxValue = reader.ReadDouble();
            Tags = new TagT[reader.ReadInt()];
            for (int i = 0; i < Tags.Length; i++)
            {
                Tags[i] = reader.ReadObject<TagT>();
            }
            mTagCounts = new BinTable<int>[reader.ReadInt()];
            for (int i = 0; i < mTagCounts.Length; i++)
            {
                mTagCounts[i] = new BinTable<int>(reader);
            }
            mTagDistrs = new BinTable<double>[reader.ReadInt()];
            for (int i = 0; i < mTagDistrs.Length; i++)
            {
                mTagDistrs[i] = new BinTable<double>(reader);
            }
            mTagIndexes = reader.LoadDictionary<TagT, int>();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }


    public class EnumTagDistrTable<TagT> : TagDistrTable<TagT>
    {
        public EnumTagDistrTable(int numOfDimensions, double binWidth, double minValue, double maxValue, params TagT[] excludedTags) :
            base(numOfDimensions, binWidth, minValue, maxValue, GetTags(excludedTags))
        {
        }

        private static TagT[] GetTags(params TagT[] excludedTags)
        {
            Preconditions.CheckArgument(typeof(TagT).IsEnum);
            return Enum.GetValues(typeof(TagT)).Cast<TagT>()
                .Where(t => excludedTags == null || !excludedTags.Any(et => et.Equals(t)))
                .ToArray();
        }
    }
}