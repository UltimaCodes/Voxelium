using OpenTK.Mathematics;
using OpenTK.Graphics.OpenGL4;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Voxelium.Graphics;
using Voxelium.World;
using System.Security.Cryptography;

namespace Voxelium.World
{
    internal class Chunk
    {
        public List<Vector3> chunkVerts;
        public List<Vector2> chunkUVs;
        public List<uint> chunkIndices;

        public const sbyte SIZE = 16;
        public const short HEIGHT = 384;
        public Vector3 position;

        public uint indexCount;

        VAO chunkVAO;
        VBO chunkVertexVBO;
        VBO chunkUVVBO;
        IBO chunkIBO;

        Texture texture;
        Block[,,] chunkBlocks = new Block[SIZE, HEIGHT, SIZE];
        public Chunk(Vector3 postition)
        {
            this.position = postition;

            chunkVerts = new List<Vector3>();
            chunkUVs = new List<Vector2>();
            chunkIndices = new List<uint>();

            float[,] heightmap = GenChunk();
            GenBlocks(heightmap);
            GenFaces(heightmap);
            BuildChunk();
        }

        public float[,] GenChunk()
        {
            float[,] heightmap = new float[SIZE, SIZE];
            
            SimplexNoise.Noise.Seed = RandomNumberGenerator.GetInt32(100000, 1000000);
            for (int x = 0; x < SIZE; x++)
            {
                for (int z = 0; z < SIZE; z++)
                {
                    heightmap[x, z] = SimplexNoise.Noise.CalcPixel2D(x, z, 0.01f);
                }
            }

            return heightmap;
        }
        public void GenBlocks(float[,] heightmap)
        {
            for (int x = 0; x < SIZE; x++)
            {
                for (int z = 0; z < SIZE; z++)
                {
                    int columnHeight = (int)(heightmap[x, z] / 10);
                    for (int y = 0; y < HEIGHT; y++)
                    {
                        BlockType type = BlockType.AIR;
                        if (y < columnHeight - 1)
                        {
                            type = BlockType.DIRT;
                        }
                        if (y == columnHeight - 1)
                        {
                            type = BlockType.GRASS;
                        }

                        chunkBlocks[x, y, z] = new Block(new Vector3(x, y, z), type);

                    }
                }
            }
        }
        public void GenFaces(float[,] heightmap)
        {
            for (int x = 0; x < SIZE; x++)
            {
                for (int z = 0; z < SIZE; z++)
                {
                    for (int y = 0; y < HEIGHT; y++)
                    {
                        int numFaces = 0;

                        if (chunkBlocks[x, y, z].type != BlockType.AIR)
                        {
                            // Left Faces
                            if (x > 0)
                            {
                                if (chunkBlocks[x - 1, y, z].type == BlockType.AIR)
                                {
                                    IntegrateFace(chunkBlocks[x, y, z], Faces.LEFT);
                                    numFaces++;
                                }
                            }
                            else
                            {
                                IntegrateFace(chunkBlocks[x, y, z], Faces.LEFT);
                                numFaces++;
                            }

                            // Right Faces
                            if (x < SIZE - 1)
                            {
                                if (chunkBlocks[x + 1, y, z].type == BlockType.AIR)
                                {
                                    IntegrateFace(chunkBlocks[x, y, z], Faces.RIGHT);
                                    numFaces++;
                                }
                            }
                            else
                            {
                                IntegrateFace(chunkBlocks[x, y, z], Faces.RIGHT);
                                numFaces++;
                            }

                            // Top Faces
                            if (y < HEIGHT - 1)
                            {
                                if (chunkBlocks[x, y + 1, z].type == BlockType.AIR)
                                {
                                    IntegrateFace(chunkBlocks[x, y, z], Faces.TOP);
                                    numFaces++;
                                }
                            }
                            else
                            {
                                IntegrateFace(chunkBlocks[x, y, z], Faces.TOP);
                                numFaces++;
                            }

                            // Bottom Faces
                            if (y > 0)
                            {
                                if (chunkBlocks[x, y - 1, z].type == BlockType.AIR)
                                {
                                    IntegrateFace(chunkBlocks[x, y, z], Faces.BOTTOM);
                                    numFaces++;
                                }
                            }
                            else
                            {
                                IntegrateFace(chunkBlocks[x, y, z], Faces.BOTTOM);
                                numFaces++;
                            }

                            // Front Faces
                            if (z < SIZE - 1)
                            {
                                if (chunkBlocks[x, y, z + 1].type == BlockType.AIR)
                                {
                                    IntegrateFace(chunkBlocks[x, y, z], Faces.FRONT);
                                    numFaces++;
                                }
                            }
                            else
                            {
                                IntegrateFace(chunkBlocks[x, y, z], Faces.FRONT);
                                numFaces++;
                            }

                            // Back Faces
                            if (z > 0)
                            {
                                if (chunkBlocks[x, y, z - 1].type == BlockType.AIR)
                                {
                                    IntegrateFace(chunkBlocks[x, y, z], Faces.BACK);
                                    numFaces++;
                                }
                            }
                            else
                            {
                                IntegrateFace(chunkBlocks[x, y, z], Faces.BACK);
                                numFaces++;
                            }

                            AddIndices(numFaces);
                        }
                    }
                }
            }
        }

        public void IntegrateFace(Block block, Faces face)
        {
            var faceData = block.GetFace(face);
            chunkVerts.AddRange(faceData.vertices);
            chunkUVs.AddRange(faceData.uv);
        }

        public void AddIndices(int amtFaces)
        {
            for (int i = 0; i < amtFaces; i++)
            {
                chunkIndices.Add(0 + indexCount);
                chunkIndices.Add(1 + indexCount);
                chunkIndices.Add(2 + indexCount);
                chunkIndices.Add(2 + indexCount);
                chunkIndices.Add(3 + indexCount);
                chunkIndices.Add(0 + indexCount);

                indexCount += 4;
            }
        }
        public void BuildChunk()
        {
            chunkVAO = new VAO();
            chunkVAO.Bind();

            chunkVertexVBO = new VBO(chunkVerts);
            chunkVertexVBO.Bind();
            chunkVAO.LinkToVAO(0, 3, chunkVertexVBO);

            chunkUVVBO = new VBO(chunkUVs);
            chunkUVVBO.Bind();
            chunkVAO.LinkToVAO(1, 2, chunkUVVBO);

            chunkIBO = new IBO(chunkIndices);

            texture = new Texture("atlas.PNG");
        } // take data and process it for rendering
        public void Render(ShaderProgram program) // drawing the chunk
        {
            program.Bind();
            chunkVAO.Bind();
            chunkIBO.Bind();
            texture.Bind();
            GL.DrawElements(PrimitiveType.Triangles, chunkIndices.Count, DrawElementsType.UnsignedInt, 0);
        }

        public void Delete()
        {
            chunkVAO.Delete();
            chunkVertexVBO.Delete();
            chunkUVVBO.Delete();
            chunkIBO.Delete();
            texture.Delete();
        }
    }
}