using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

public class QueryProbe : MonoBehaviour {
	public Color Color;
	public float Speed = 0.1f;
	
	public static List<QueryProbe> All = new List<QueryProbe>();

	float3 m_posSeed;
	
	void OnEnable() {
		GetComponent<MeshRenderer>().material.color = Color;
		m_posSeed = Random.insideUnitSphere;

		All.Add(this);
	}

	void OnDisable() {
		All.Remove(this);
	}

	// Update is called once per frame
    void Update() {
	    noise.snoise((float3)transform.position + m_posSeed * Time.time, out float3 grad);
	    transform.position += (Vector3) grad * Time.deltaTime * Speed;
    }
}
