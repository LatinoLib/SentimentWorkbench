using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System.IO;
using Latino.Model;
using Latino.TextMining;
using TextTask;
using TextTask.DataSource;

namespace OrdinalReg.DataSource
{
    public class GeneratedSource : CsvLabeledTextSource
    {
        public GeneratedSource(string fileName) : base(fileName)
        {
            Language = GetLanguage();
        }

        public Language GetLanguage()
        {
            string name = Path.GetFileNameWithoutExtension(FileName).Split('_')[1];
            switch (name)
            {
                case "English":
                    return Language.English;
                case "Russia":
                    return Language.Russian;
                case "Kazahstan":
                    return Language.Unspecified;
                case "Montenegro":
                    return Language.Serbian;
                case "Portugal":
                    return Language.Portuguese;
                case "Serbia":
                    return Language.Serbian;
                case "Bosnia":
                    return Language.Serbian;
                case "Germany":
                    return Language.German;
                case "Slovenia":
                    return Language.Slovene;
                case "Saudi Arabia":
                    return Language.Unspecified;
                case "Croatia":
                    return Language.Serbian;
                case "Bulgaria":
                    return Language.Bulgarian;
                case "Spain":
                    return Language.Spanish;
                case "Hungary":
                    return Language.Hungarian;
                case "Poland":
                    return Language.Polish;
                case "Sweden":
                    return Language.Swedish;
                case "Slovakia":
                    return Language.Slovak;
                case "Argentina":
                    return Language.Spanish;
                case "Albania":
                    return Language.Unspecified;
                case "SerboCroatian":
                    return Language.Serbian;
                case "France":
                    return Language.French;
                case "Italy":
                    return Language.Italian;
                case "Nederlands":
                    return Language.Dutch;
                case "Turkey":
                    return Language.Turkish;

                default:
                    throw new 
                        NotImplementedException();
            }
        }

        public static TweetDataSet[] GetFromDirectory(string path)
        {
            return Directory.GetFiles(path)
                .GroupBy(fn => Path.GetFileNameWithoutExtension(fn).Split(new[] { '_' }, 2)[1])
                .Select(g => 
                    {
                        var tds = new TweetDataSet();
                        foreach (string fn in g)
                        {
                            if (fn.Contains("test"))
                            {
                                tds.Test = new GeneratedSource(fn);
                            }
                            else if (fn.Contains("train"))
                            {
                                tds.Train = new GeneratedSource(fn);
                            }
                        }
                        return tds;
                    })
                .ToArray();
        }

        private Tweet mPrevTweet;
        private readonly List<SentimentLabel> mPrevLabels = new List<SentimentLabel>();

        protected override bool LoadLabeledTweet(string[] fields, out SentimentLabel label, out Tweet tweet)
        {
            tweet = new Tweet
                {
                    Id = UInt64.Parse(fields[0]),
                    Date = DateTime.ParseExact(fields[1], "yyyy-MM-dd HH:mm:ss", CultureInfo.CreateSpecificCulture("en-US")),
                    User = fields[2],
                    Text = fields[3],
                    AgreementRank = 1
                };
            switch (fields[4])
            {
                case "Negative": label = SentimentLabel.Negative; break;
                case "Neutral": label = SentimentLabel.Neutral; break;
                case "Positive": label = SentimentLabel.Positive; break;
                case "Skip":
                case "Exclude":
                    label = default(SentimentLabel);
                    return false;

                default: throw new Exception("invalid value");
            }

            if (mPrevTweet != null && mPrevTweet.Id == tweet.Id)
            {
                mPrevLabels.Add(label);
                mPrevTweet.DisagreementRank = mPrevLabels.Max() - mPrevLabels.Min();
                mPrevTweet.AgreementRank = mPrevLabels.Distinct().Count() == 1 ? mPrevLabels.Count : 0;
                
                label = LabelFromVoting(mPrevLabels);
                tweet = null;

                return true;
            }
            
            mPrevLabels.Clear();
            mPrevLabels.Add(label);
            mPrevTweet = tweet;
            return true;
        }


        protected SentimentLabel LabelFromVoting(List<SentimentLabel> allLabels)
        {
            int sum = allLabels.Sum(l => (int)l);
            return sum == 0 ? SentimentLabel.Neutral : (sum > 0 ? SentimentLabel.Positive : SentimentLabel.Negative);
        }
    }
}