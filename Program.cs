using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Training
{
    internal sealed class Sample
    {
        public int Epoch { get; set; }
        public double ValidLoss { get; set; }
        public double MoveAccuracy { get; set; }
    }

    internal class Program
    {
        private static readonly Regex ProgressRegex = new Regex(
            @"^PROGRESS \(calc_loss\):.*?\bepoch\s+(\d+)\s*$",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex ValidLossRegex = new Regex(
            @"^\s*-\s*valid loss\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        private static readonly Regex MoveAccuracyRegex = new Regex(
            @"^\s*-\s*move accuracy\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%\s*$",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        static void Main(string[] args)
        {
            try
            {
                if (args.Length != 1)
                {
                    Console.WriteLine("Usage: Training.exe <logfile>");
                    return;
                }

                string fileName = args[0];

                if (!File.Exists(fileName))
                {
                    Console.Error.WriteLine("File not found: " + fileName);
                    return;
                }

                List<Sample> samples = ReadSamples(fileName);

                if (samples.Count == 0)
                {
                    Console.Error.WriteLine("No valid PROGRESS blocks found.");
                    return;
                }

                samples = samples.OrderBy(s => s.Epoch).ToList();

                string outputFile = Path.Combine(
                    Path.GetDirectoryName(Path.GetFullPath(fileName)) ?? ".",
                    "plot_progress.wl");

                string runName = Path.GetFileNameWithoutExtension(fileName);
                File.WriteAllText(outputFile, BuildMathematica(samples, runName), new UTF8Encoding(false));

                Console.WriteLine("Read {0} samples.", samples.Count);
                Console.WriteLine("Wrote Mathematica file: {0}", outputFile);
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(e.Message);
            }
        }

        private static List<Sample> ReadSamples(string fileName)
        {
            List<Sample> samples = new List<Sample>();

            int? currentEpoch = null;
            double? currentValidLoss = null;
            double? currentMoveAccuracy = null;

            foreach (string line in File.ReadLines(fileName))
            {
                Match progressMatch = ProgressRegex.Match(line);
                if (progressMatch.Success)
                {
                    if (currentEpoch.HasValue &&
                        currentValidLoss.HasValue &&
                        currentMoveAccuracy.HasValue)
                    {
                        samples.Add(new Sample
                        {
                            Epoch = currentEpoch.Value,
                            ValidLoss = currentValidLoss.Value,
                            MoveAccuracy = currentMoveAccuracy.Value
                        });
                    }

                    currentEpoch = int.Parse(progressMatch.Groups[1].Value, CultureInfo.InvariantCulture);
                    currentValidLoss = null;
                    currentMoveAccuracy = null;
                    continue;
                }

                if (currentEpoch.HasValue)
                {
                    Match validLossMatch = ValidLossRegex.Match(line);
                    if (validLossMatch.Success)
                    {
                        currentValidLoss = double.Parse(
                            validLossMatch.Groups[1].Value,
                            CultureInfo.InvariantCulture);
                        continue;
                    }

                    Match moveAccuracyMatch = MoveAccuracyRegex.Match(line);
                    if (moveAccuracyMatch.Success)
                    {
                        currentMoveAccuracy = double.Parse(
                            moveAccuracyMatch.Groups[1].Value,
                            CultureInfo.InvariantCulture);
                    }
                }
            }

            if (currentEpoch.HasValue &&
                currentValidLoss.HasValue &&
                currentMoveAccuracy.HasValue)
            {
                samples.Add(new Sample
                {
                    Epoch = currentEpoch.Value,
                    ValidLoss = currentValidLoss.Value,
                    MoveAccuracy = currentMoveAccuracy.Value
                });
            }

            return samples;
        }
        private static Sample GetTrendSupportedSample(List<Sample> samples, int smoothWindow)
        {
            if (samples == null || samples.Count == 0)
                throw new ArgumentException("No samples supplied.", nameof(samples));

            if (smoothWindow < 3)
                smoothWindow = 3;

            if (smoothWindow % 2 == 0)
                smoothWindow--;

            if (samples.Count < smoothWindow)
            {
                return samples
                    .OrderBy(s => s.ValidLoss)
                    .ThenBy(s => s.Epoch)
                    .First();
            }

            var smoothPoints = new List<(double Epoch, double ValidLoss)>();

            for (int i = 0; i <= samples.Count - smoothWindow; i++)
            {
                List<Sample> window = samples.Skip(i).Take(smoothWindow).ToList();

                double avgEpoch = window.Average(s => (double)s.Epoch);
                double avgValidLoss = window.Average(s => s.ValidLoss);

                smoothPoints.Add((avgEpoch, avgValidLoss));
            }

            double smoothMinEpoch = smoothPoints
                .OrderBy(p => p.ValidLoss)
                .ThenBy(p => p.Epoch)
                .First()
                .Epoch;

            return samples
                .Where(s => s.Epoch <= smoothMinEpoch)
                .OrderBy(s => s.ValidLoss)
                .ThenBy(s => s.Epoch)
                .First();
        }

        private static string BuildMathematica(List<Sample> samples, string runName)
        {
            Sample best = samples
                .OrderBy(s => s.ValidLoss)
                .ThenBy(s => s.Epoch)
                .First();

            const int smoothWindow = 101;

            Sample adopted = GetTrendSupportedSample(samples, smoothWindow);

            bool adoptedDiffers =
                adopted.Epoch != best.Epoch ||
                Math.Abs(adopted.ValidLoss - best.ValidLoss) > 1e-15;

            string bestEpochText = best.Epoch.ToString(CultureInfo.InvariantCulture);
            string bestVLossText = best.ValidLoss.ToString("F7", CultureInfo.InvariantCulture);
            string bestSummaryText = "Lowest validation loss: " + bestVLossText + " at epoch " + bestEpochText + ".";
            string bestPlotLabelText = "min = " + bestVLossText + " at epoch " + bestEpochText;

            string adoptedEpochText = adopted.Epoch.ToString(CultureInfo.InvariantCulture);
            string adoptedVLossText = adopted.ValidLoss.ToString("F7", CultureInfo.InvariantCulture);
            string adoptedSummaryText = "Trend-supported validation loss: " + adoptedVLossText + " at epoch " + adoptedEpochText + ".";
            string adoptedPlotLabelText = "trend = " + adoptedVLossText + " at epoch " + adoptedEpochText;

            int firstEpoch = samples.First().Epoch;
            int lastEpoch = samples.Last().Epoch;

            StringBuilder sb = new StringBuilder();

            sb.AppendLine("ClearAll[vloss, macc, bestEpoch, bestVLoss, adoptEpoch, adoptVLoss, epochMin, epochMax, smoothWindow, vlossSmooth, p1Base, p1Fit, p1, p2, nb, nbFile, scriptDir];");
            sb.AppendLine();
            sb.AppendLine("scriptDir = If[");
            sb.AppendLine("  StringLength[$InputFileName] > 0,");
            sb.AppendLine("  DirectoryName[ExpandFileName[$InputFileName]],");
            sb.AppendLine("  Directory[]");
            sb.AppendLine("];");
            sb.AppendLine();

            sb.Append("vloss = {");
            for (int i = 0; i < samples.Count; i++)
            {
                if (i > 0)
                {
                    sb.Append(", ");
                }

                sb.Append("{");
                sb.Append(samples[i].Epoch.ToString(CultureInfo.InvariantCulture));
                sb.Append(", ");
                sb.Append(samples[i].ValidLoss.ToString("G17", CultureInfo.InvariantCulture));
                sb.Append("}");
            }
            sb.AppendLine("};");

            sb.Append("macc = {");
            for (int i = 0; i < samples.Count; i++)
            {
                if (i > 0)
                {
                    sb.Append(", ");
                }

                sb.Append("{");
                sb.Append(samples[i].Epoch.ToString(CultureInfo.InvariantCulture));
                sb.Append(", ");
                sb.Append(samples[i].MoveAccuracy.ToString("G17", CultureInfo.InvariantCulture));
                sb.Append("}");
            }
            sb.AppendLine("};");
            sb.AppendLine();

            sb.Append("bestEpoch = ");
            sb.Append(bestEpochText);
            sb.AppendLine(";");

            sb.Append("bestVLoss = ");
            sb.Append(best.ValidLoss.ToString("G17", CultureInfo.InvariantCulture));
            sb.AppendLine(";");
            sb.AppendLine();

            sb.Append("adoptEpoch = ");
            sb.Append(adoptedEpochText);
            sb.AppendLine(";");

            sb.Append("adoptVLoss = ");
            sb.Append(adopted.ValidLoss.ToString("G17", CultureInfo.InvariantCulture));
            sb.AppendLine(";");
            sb.AppendLine();

            sb.Append("epochMin = ");
            sb.Append(firstEpoch.ToString(CultureInfo.InvariantCulture));
            sb.AppendLine(";");

            sb.Append("epochMax = ");
            sb.Append(lastEpoch.ToString(CultureInfo.InvariantCulture));
            sb.AppendLine(";");
            sb.AppendLine();

            sb.AppendLine("p1Base = ListLinePlot[");
            sb.AppendLine("  vloss,");
            sb.AppendLine("  Frame -> True,");
            sb.AppendLine("  FrameLabel -> {\"epoch\", \"valid loss\"},");
            sb.AppendLine("  PlotLabel -> Style[\"Validation Loss vs Epoch\", 16, Bold],");
            sb.AppendLine("  PlotStyle -> Directive[Blue, Thick],");
            sb.AppendLine("  GridLines -> Automatic,");
            sb.AppendLine("  GridLinesStyle -> Directive[GrayLevel[0.85]],");
            sb.AppendLine("  ImageSize -> {1300, 450},");
            sb.AppendLine("  BaseStyle -> {FontFamily -> \"Arial\", FontSize -> 13},");
            sb.AppendLine("  Epilog -> {");
            sb.AppendLine("    Red, AbsolutePointSize[8], Point[{bestEpoch, bestVLoss}],");
            sb.AppendLine("    Inset[");
            sb.AppendLine("      Framed[");
            sb.Append("        Style[\"");
            sb.Append(bestPlotLabelText);
            sb.AppendLine("\", 12],");
            sb.AppendLine("        Background -> White,");
            sb.AppendLine("        FrameMargins -> Tiny");
            sb.AppendLine("      ],");
            sb.AppendLine("      Offset[{140, 18}, {bestEpoch, bestVLoss}]");
            sb.AppendLine("    ]");

            if (adoptedDiffers)
            {
                sb.AppendLine("    ,Darker[Green], AbsolutePointSize[8], Point[{adoptEpoch, adoptVLoss}],");
                sb.AppendLine("    Inset[");
                sb.AppendLine("      Framed[");
                sb.Append("        Style[\"");
                sb.Append(adoptedPlotLabelText);
                sb.AppendLine("\", 12],");
                sb.AppendLine("        Background -> White,");
                sb.AppendLine("        FrameMargins -> Tiny");
                sb.AppendLine("      ],");
                sb.AppendLine("      Offset[{140, -18}, {adoptEpoch, adoptVLoss}]");
                sb.AppendLine("    ]");
            }

            sb.AppendLine("  }");
            sb.AppendLine("];");
            sb.AppendLine();

            sb.Append("smoothWindow = ");
            sb.Append(smoothWindow.ToString(CultureInfo.InvariantCulture));
            sb.AppendLine(";");

            sb.AppendLine("vlossSmooth = Transpose[{");
            sb.AppendLine("  MovingAverage[vloss[[All, 1]], smoothWindow],");
            sb.AppendLine("  MovingAverage[vloss[[All, 2]], smoothWindow]");
            sb.AppendLine("}];");

            sb.AppendLine("p1Fit = ListLinePlot[");
            sb.AppendLine("  vlossSmooth,");
            sb.AppendLine("  PlotStyle -> Directive[Orange, Dashed, Thick]");
            sb.AppendLine("];");
            sb.AppendLine();

            sb.AppendLine("p1 = Show[p1Base, p1Fit];");
            sb.AppendLine();

            sb.AppendLine("p2 = ListLinePlot[");
            sb.AppendLine("  macc,");
            sb.AppendLine("  Frame -> True,");
            sb.AppendLine("  FrameLabel -> {\"epoch\", \"move accuracy (%)\"},");
            sb.AppendLine("  PlotLabel -> Style[\"Move Accuracy vs Epoch\", 16, Bold],");
            sb.AppendLine("  PlotStyle -> Directive[DarkGray, Thick],");
            sb.AppendLine("  GridLines -> Automatic,");
            sb.AppendLine("  GridLinesStyle -> Directive[GrayLevel[0.85]],");
            sb.AppendLine("  ImageSize -> {1300, 450},");
            sb.AppendLine("  BaseStyle -> {FontFamily -> \"Arial\", FontSize -> 13}");
            sb.AppendLine("];");
            sb.AppendLine();

            sb.AppendLine("nb = CreateDocument[");
            sb.AppendLine("  {");
            string notebookTitle = "Training Progress Plots " + runName;
            sb.Append("    Cell[\"");
            sb.Append(notebookTitle);
            sb.AppendLine("\", \"Section\"],");
            sb.AppendLine("    Cell[\"Generated from training log.\", \"Text\"],");

            sb.Append("    Cell[\"");
            sb.Append(bestSummaryText);
            sb.AppendLine("\", \"Text\"],");

            sb.Append("    Cell[\"");
            sb.Append(adoptedSummaryText);
            sb.AppendLine("\", \"Text\"],");

            sb.AppendLine("    ExpressionCell[p1],");
            sb.AppendLine("    ExpressionCell[p2]");
            sb.AppendLine("  },");

            sb.Append("  WindowTitle -> \"");
            sb.Append(notebookTitle);
            sb.AppendLine("\",");

            sb.AppendLine("  Visible -> True");
            sb.AppendLine("];");
            sb.AppendLine();

            sb.AppendLine("nbFile = FileNameJoin[{scriptDir, \"plot_progress.nb\"}];");
            sb.AppendLine("NotebookSave[nb, nbFile];");
            sb.AppendLine("Print[\"Wrote notebook to: \" <> nbFile];");
            sb.AppendLine("nb");

            return sb.ToString();
        }
    }
}