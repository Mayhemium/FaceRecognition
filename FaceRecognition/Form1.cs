using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace FaceRecognition
{
    public partial class Form1 : Form
    {
        MCvFont font = new MCvFont(Emgu.CV.CvEnum.FONT.CV_FONT_HERSHEY_TRIPLEX,0.6d,0.6d);
        HaarCascade faceDetectionHaarCascade;
        Image<Bgr, byte> Frame;
        Capture camera;
        Image<Gray, byte> result;
        Image<Gray, byte> trainedFace = null;
        Image<Gray, byte> grayFace = null;
        List<Image<Gray, byte>> trainingImages = new List<Image<Gray, byte>>();
        List<string> labels = new List<string>();
        int count, numLabels;
        string name;

        public Form1()
        {
            InitializeComponent();

            faceDetectionHaarCascade = new HaarCascade("haarcascade_frontalface_default.xml");

            try
            {
                string labelsinf = File.ReadAllText(Application.StartupPath + "/faces/faces.txt");
                string[] Labels = labelsinf.Split(',');
                numLabels = Convert.ToInt16(Labels[0]);
                string facesLoad;
                for(int i = 1; i < numLabels + 1; i++)
                {
                    facesLoad = "face" + i + ".bmp";
                    trainingImages.Add(new Image<Gray, byte>(Application.StartupPath + "/faces/" + facesLoad));
                    labels.Add(Labels[i]);
                }

            }catch(Exception e)
            {
                MessageBox.Show(e.Message);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            camera = new Capture();
            Application.Idle += new EventHandler(FrameProcedure);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            count++;
            grayFace = camera.QueryGrayFrame().Resize(500, 500, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            MCvAvgComp[][] detectedFaces = grayFace.DetectHaarCascade(faceDetectionHaarCascade, 1.2, 10, Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING, new Size(20, 20));
            foreach (MCvAvgComp f in detectedFaces[0])
            {
                trainedFace = Frame.Copy(f.rect).Convert<Gray, byte>();
                break;
            }
            trainedFace = result.Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            trainingImages.Add(trainedFace);
            labels.Add(textBox1.Text);
            File.WriteAllText(Application.StartupPath + "/faces/faces.txt", trainingImages.ToArray().Length.ToString() + ",");
            for (int i = 1; i < trainingImages.ToArray().Length + 1; i++)
            {
                trainingImages.ToArray()[i - 1].Save(Application.StartupPath + "/faces/face" + i + ".bmp");
                File.AppendAllText(Application.StartupPath + "/faces/faces.txt", labels.ToArray()[i - 1] + ",");
            }
            MessageBox.Show(textBox1.Text + " added");
        }

        private void FrameProcedure(object sender, EventArgs e)
        {
            Frame = camera.QueryFrame().Resize(500, 500, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            grayFace = Frame.Convert<Gray, Byte>();
            MCvAvgComp[][] facesDetected = grayFace.DetectHaarCascade(faceDetectionHaarCascade,1.2,10,Emgu.CV.CvEnum.HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,new Size(20,20));
            foreach (MCvAvgComp f in facesDetected[0])
            {
                name = "";
                result = Frame.Copy(f.rect).Convert<Gray, byte>().Resize(100,100,Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                Frame.Draw(f.rect,new Bgr(Color.Green),3);
                if(trainingImages.ToArray().Length != 0)
                {
                    MCvTermCriteria termCriteria = new MCvTermCriteria(count,0.001);
                    EigenObjectRecognizer recognizer = new EigenObjectRecognizer(trainingImages.ToArray(),labels.ToArray(),1500,ref termCriteria);
                    name = recognizer.Recognize(result);
                    Frame.Draw(name, ref font, new Point(f.rect.X - 2, f.rect.Y - 2), new Bgr(Color.Red));
                }
            }
            imageBox1.Image = Frame;

        }
    }
}
