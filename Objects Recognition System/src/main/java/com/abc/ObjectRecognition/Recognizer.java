package com.abc.ObjectRecognition;

import java.io.File;
import java.io.IOException;

import java.awt.Image;
import java.awt.Dimension;
import java.awt.FlowLayout;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.charset.Charset;

import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import javax.imageio.ImageIO;

import javax.swing.JLabel;
import javax.swing.JFrame;
import javax.swing.JButton;
import javax.swing.ImageIcon;
import javax.swing.JTextField;
import javax.swing.JFileChooser;
import javax.swing.SwingUtilities;
import javax.swing.AbstractAction;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Tensor;
import org.tensorflow.Session;
import org.tensorflow.DataType;

import com.github.sarxos.webcam.Webcam;
import com.github.sarxos.webcam.WebcamPanel;
import com.github.sarxos.webcam.WebcamResolution;
import com.esotericsoftware.tablelayout.swing.Table;


public class Recognizer extends JFrame implements ActionListener {
    private final JButton incep= new JButton("Take A Picture");
    final Table table =new Table();
    final JButton predict = new JButton("Predict");
    final JButton img= new JButton("Choose Image");
    final JLabel viewer = new JLabel();
    final JFileChooser imgch= new JFileChooser();
    final JTextField result=new JTextField();
    final JTextField imgpth;
    private JLabel label;
    
    
    final FileNameExtensionFilter imgfilter = new FileNameExtensionFilter(
            "JPG & JPEG Images", "jpg", "jpeg");
    private String modelpath;
    private String imagepath;
    private boolean modelselected = false;
    private byte[] graphDef;
    private List<String> labels;


    public Recognizer() {
       setTitle("Object Recognition System");
       setSize(500, 500);
       
        predict.setEnabled(false);
 
        incep.addActionListener(this);
        img.addActionListener(this);
        predict.addActionListener(this);
        
        imgch.setFileFilter(imgfilter);
        imgch.setFileSelectionMode(JFileChooser.FILES_ONLY);
      
        imgpth = new JTextField();
        imgpth.setEditable(false);
        
        getContentPane().add(table);
        
        table.addCell(incep).colspan(2);
        table.row();
        table.addCell(imgpth).width(250);
        table.addCell(img);
        table.row();
        table.addCell(viewer).size(300, 300).colspan(2);
        table.row();
        table.addCell(predict).colspan(2);
        table.row();
        table.addCell(result).width(300).colspan(2);
        table.row();
        
        label = new JLabel();
        label.setIcon(new ImageIcon("C:\\Users\\Asus\\Downloads\\Background.jpg"));
        label.setBounds(0,0,500,500);
        label.setHorizontalAlignment(JLabel.CENTER);
        label.setVerticalAlignment(JLabel.CENTER);
        table.setLayout(null);
        table.add(label);

        setLocationRelativeTo(null);
        setResizable(false);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
  
        setVisible(true);

    }
   

    @Override
    public void actionPerformed(ActionEvent e)  {


                File files = new File("C:\\Users\\Asus\\Downloads\\Training Model");
                modelpath = files.getAbsolutePath();
         
                System.out.println("Opening: " + files.getAbsolutePath());
                modelselected = true;
                graphDef = readAllBytesOrExit(Paths.get(modelpath, "tensorflow_inception_graph.pb"));
                labels = readAllLinesOrExit(Paths.get(modelpath, "imagenet_comp_graph_label_strings.txt"));
           
                if (e.getSource()==incep){
                   
                    SnapMeAction snapMeAction= new SnapMeAction();
                    final JFrame window = new JFrame("Webcam");
		
		
                    for (Webcam webcam : webcams) {
                       
                    webcam.setViewSize(size);
                    WebcamPanel panel = new WebcamPanel(webcam, size,false);
                                           
			panel.setFPSDisplayed(true);
                        panel.setMirrored(true);
			panels.add(panel);
                        add(panel);  
                        window.add(panel);
                btSnapMe.setEnabled(false);
		btStop.setEnabled(false);

		setLayout(new FlowLayout());
                panel.add(btSnapMe);
		panel.add(btStart);
                panel.add(btStop);
		
                   window.setResizable(true);
	
		window.pack();
		window.setVisible(true);
               
		}
       
           }
                else if (e.getSource() == img) {
             
            int returnVal = imgch.showOpenDialog(Recognizer.this);
            if (returnVal == JFileChooser.APPROVE_OPTION) {
                try {
                    File file = imgch.getSelectedFile();
                    imagepath = file.getAbsolutePath();
                    imgpth.setText(imagepath);
                    System.out.println("Image Path: " + imagepath);
                    Image imge = ImageIO.read(file);

                    viewer.setIcon(new ImageIcon(imge.getScaledInstance(300, 300, 300)));
                    if (modelselected) {
                        predict.setEnabled(true);
                    }
                } catch (IOException ex) {
                    Logger.getLogger(Recognizer.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else {
                System.out.println("Process was cancelled by user.");
            }
        } else if (e.getSource() == predict) {
            byte[] imageBytes = readAllBytesOrExit(Paths.get(imagepath));

            try (Tensor image = Tensor.create(imageBytes)) {
                float[] labelProbabilities = executeInceptionGraph(graphDef, image);
                int bestLabelIdx = maxIndex(labelProbabilities);
                result.setText("");
                result.setText(String.format(
                                "BEST MATCH: %s (%.2f%% likely)",
                                labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
                System.out.println(
                        String.format(
                                "BEST MATCH: %s (%.2f%% likely)",
                                labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
            }

        }
    }

 public class SnapMeAction extends AbstractAction {
        
		public SnapMeAction() {
			super("Snapshot");
                        
                        
		}

		@Override
		public void actionPerformed(ActionEvent e) {
			try {
				for (int i = 0; i < webcams.size(); i++) {
					Webcam webcam = webcams.get(i);
					File file = new File(String.format("test-%d.jpg", i));
					ImageIO.write(webcam.getImage(), "JPG", file);
					System.out.format("Image for %s saved in %s \n", webcam.getName(), file);
				}
			} catch (IOException e1) {
			}
		}
	}

	private class StartAction extends AbstractAction implements Runnable {

		public StartAction() {
			super("Start");
		}

		@Override
		public void actionPerformed(ActionEvent e) {

			btStart.setEnabled(false);
			btSnapMe.setEnabled(true);

			// remember to start panel asynchronously - otherwise GUI will be
			// blocked while OS is opening webcam HW (will have to wait for
			// webcam to be ready) and this causes GUI to hang, stop responding
			// and repainting

			executor.execute(this);
		}

		@Override
		public void run() {

			btStop.setEnabled(true);

			for (WebcamPanel panel : panels) {
				panel.start();
			}
		}
	}

	private class StopAction extends AbstractAction {

		public StopAction() {
			super("Stop");
		}

		@Override
		public void actionPerformed(ActionEvent e) {

			btStart.setEnabled(true);
			btSnapMe.setEnabled(false);
			btStop.setEnabled(false);

			for (WebcamPanel panel : panels) {
				panel.stop();
			}
		}
	}

	private final Executor executor = Executors.newSingleThreadExecutor();

	
	private final Dimension size = WebcamResolution.VGA.getSize();

	private final List<Webcam> webcams = Webcam.getWebcams();
	private final List<WebcamPanel> panels = new ArrayList<>();

	private final JButton btSnapMe = new JButton(new SnapMeAction());
	private final JButton btStart = new JButton(new StartAction());
	private final JButton btStop = new JButton(new StopAction());

	

	
    private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                    Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("softmax").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                return result.copyTo(new float[1][nlabels])[0];
            }
        }
    }

    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }

    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }

    // In the fullness of time, equivalents of the methods of this class should be auto-generated from
    // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
    // like Python, C++ and Go.
    static class GraphBuilder {

        GraphBuilder(Graph g) {
            this.g = g;
        }

        Output div(Output x, Output y) {
            return binaryOp("Div", x, y);
        }

        Output sub(Output x, Output y) {
            return binaryOp("Sub", x, y);
        }

        Output resizeBilinear(Output images, Output size) {
            return binaryOp("ResizeBilinear", images, size);
        }

        Output expandDims(Output input, Output dim) {
            return binaryOp("ExpandDims", input, dim);
        }

        Output cast(Output value, DataType dtype) {
            return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0);
        }

        Output decodeJpeg(Output contents, long channels) {
            return g.opBuilder("DecodeJpeg", "DecodeJpeg")
                    .addInput(contents)
                    .setAttr("channels", channels)
                    .build()
                    .output(0);
        }

        Output constant(String name, Object value) {
            try (Tensor t = Tensor.create(value)) {
                return g.opBuilder("Const", name)
                        .setAttr("dtype", t.dataType())
                        .setAttr("value", t)
                        .build()
                        .output(0);
            }
        }

        private Output binaryOp(String type, Output in1, Output in2) {
            return g.opBuilder(type, type).addInput(in1).addInput(in2).build().output(0);
        }

        private final Graph g;
    }

    public static void main(String[] args) {

        SwingUtilities.invokeLater(() -> {
            new Recognizer().setVisible(true);
            
        });
    }

}