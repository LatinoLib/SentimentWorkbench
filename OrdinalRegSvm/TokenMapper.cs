using System;
using System.Collections.Generic;
using System.IO;
using Latino;
using Newtonsoft.Json.Linq;

namespace OrdinalReg
{
    public class TokenMapper
    {
        private readonly Dictionary<string, string> mTokenTags = new Dictionary<string, string>();

        public TokenMapper(string[] tags)
        {
            Tags = Preconditions.CheckNotNull(tags);
        }

        public string[] Tags { get; private set; }

        public void Add(string token, int tagIdx)
        {
            Preconditions.CheckArgument(!string.IsNullOrEmpty(token));
            Preconditions.CheckArgument(tagIdx >= 0 && tagIdx < Tags.Length);
            mTokenTags.Add(token, Tags[tagIdx]);
        }

        public string Get(string token)
        {
            string tag;
            return mTokenTags.TryGetValue(token, out tag) ? tag : null;
        }
    }

    public class HappinessMapper : TokenMapper
    {
        public HappinessMapper(string jsonFileName) : base(new[]
            {
                "happiness_1", "happiness_2", "happiness_3", "happiness_4", "happiness_5", 
                "happiness_6", "happiness_7", "happiness_8", "happiness_9", "happiness_10"
            })
        {
            JObject happinessWordsJson = JObject.Parse(File.ReadAllText(jsonFileName));
            foreach (JToken jtok in happinessWordsJson.Value<JArray>("objects"))
            {
                string token = jtok.Value<string>("word");
                double happiness = jtok.Value<double>("happs");
                int tagIdx = (int)Math.Round(happiness, MidpointRounding.AwayFromZero);
                Add(token, tagIdx);
            }
        }
    }
}