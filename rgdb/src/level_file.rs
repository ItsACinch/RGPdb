//! Level file format v2: header, nodes, CSR edges, room map, rooms, relation vocab.

use crate::graph::{Graph, NodeProps, EdgeProps, RelationId};
use crate::rooms::RoomCollection;
use crate::relation::RelationVocab;
use memmap2::MmapOptions;
use std::fs::File;
use std::io::{Read, Write, Seek, SeekFrom};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use thiserror::Error;

const MAGIC: &[u8; 4] = b"RGDB";
const VERSION: u32 = 2;
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum LevelFileError {
    #[error("Invalid magic number")]
    InvalidMagic,
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Structure mismatch: {0}")]
    StructureMismatch(String),
}

fn write_node<W: Write>(w: &mut W, p: &NodeProps) -> std::io::Result<()> {
    w.write_all(&p.reflection.to_le_bytes())?;
    w.write_all(&p.refraction_index.to_le_bytes())?;
    Ok(())
}

fn read_node<R: Read>(r: &mut R) -> std::io::Result<NodeProps> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    let reflection = f32::from_le_bytes(b);
    r.read_exact(&mut b)?;
    let refraction_index = f32::from_le_bytes(b);
    Ok(NodeProps { reflection, refraction_index })
}

fn write_edge<W: Write>(w: &mut W, e: &EdgeProps) -> std::io::Result<()> {
    w.write_all(&e.attenuation.to_le_bytes())?;
    w.write_u16::<LittleEndian>(e.relation)?;
    w.write_u8(if e.is_portal { 1 } else { 0 })?;
    w.write_u8(0)?; // pad to 8 bytes
    Ok(())
}

fn read_edge<R: Read>(r: &mut R) -> std::io::Result<EdgeProps> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    let attenuation = f32::from_le_bytes(b);
    let relation = r.read_u16::<LittleEndian>()?;
    let is_portal = r.read_u8()? != 0;
    let _pad = r.read_u8()?;
    Ok(EdgeProps { attenuation, relation, is_portal })
}

/// Write graph + rooms + relation vocab to a v2 level file.
pub fn write_level_file(
    graph: &Graph,
    rooms: &RoomCollection,
    vocab: &RelationVocab,
    path: &str,
) -> Result<(), LevelFileError> {
    let mut file = File::create(path)?;

    // Header: magic, version, num_nodes, num_edges (all u32) = 16 bytes.
    file.write_all(MAGIC)?;
    file.write_u32::<LittleEndian>(VERSION)?;
    file.write_u32::<LittleEndian>(graph.num_nodes() as u32)?;
    file.write_u32::<LittleEndian>(graph.num_edges() as u32)?;

    // Nodes.
    for p in graph.node_props() {
        write_node(&mut file, p)?;
    }
    // CSR: row_ptr (u32), col_idx (u32), edge_props (8 bytes).
    for &ptr in graph.row_ptr() {
        file.write_u32::<LittleEndian>(ptr as u32)?;
    }
    for &c in graph.col_idx() {
        file.write_u32::<LittleEndian>(c)?;
    }
    for e in graph.edge_props() {
        write_edge(&mut file, e)?;
    }
    // Room map.
    file.write_u32::<LittleEndian>(graph.room_map().len() as u32)?;
    for &r in graph.room_map() {
        file.write_u32::<LittleEndian>(r)?;
    }
    // Rooms.
    file.write_u32::<LittleEndian>(rooms.num_rooms() as u32)?;
    for room in &rooms.rooms {
        file.write_u32::<LittleEndian>(room.id)?;
        file.write_u32::<LittleEndian>(room.node_start as u32)?;
        file.write_u32::<LittleEndian>(room.node_count as u32)?;
    }
    // Relation vocab: count, then each name (len-prefixed utf8), then n*n f32 matrix.
    let n = vocab.len();
    file.write_u32::<LittleEndian>(n as u32)?;
    for i in 0..n {
        let name = vocab.name(i as RelationId).unwrap_or("");
        let bytes = name.as_bytes();
        file.write_u32::<LittleEndian>(bytes.len() as u32)?;
        file.write_all(bytes)?;
    }
    for a in 0..n {
        for b in 0..n {
            file.write_all(&vocab.similarity(a as RelationId, b as RelationId).to_le_bytes())?;
        }
    }
    Ok(())
}

/// Read a v2 level file (streamed over an mmap).
pub fn read_level_file_mmap(
    path: &str,
) -> Result<(Graph, RoomCollection, RelationVocab), LevelFileError> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    if size < 16 {
        return Err(LevelFileError::StructureMismatch("file too small".into()));
    }
    if size > MAX_FILE_SIZE {
        return Err(LevelFileError::StructureMismatch("file too large".into()));
    }
    let mmap = unsafe { MmapOptions::new().len(size as usize).map(&file)? };
    let mut cur = &mmap[..];

    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(LevelFileError::InvalidMagic);
    }
    let version = cur.read_u32::<LittleEndian>()?;
    if version != VERSION {
        return Err(LevelFileError::UnsupportedVersion(version));
    }
    let num_nodes = cur.read_u32::<LittleEndian>()? as usize;
    let num_edges = cur.read_u32::<LittleEndian>()? as usize;

    let mut node_props = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        node_props.push(read_node(&mut cur)?);
    }
    let mut row_ptr = Vec::with_capacity(num_nodes + 1);
    for _ in 0..=num_nodes {
        row_ptr.push(cur.read_u32::<LittleEndian>()? as usize);
    }
    let mut col_idx = Vec::with_capacity(num_edges);
    for _ in 0..num_edges {
        col_idx.push(cur.read_u32::<LittleEndian>()?);
    }
    let mut edge_props = Vec::with_capacity(num_edges);
    for _ in 0..num_edges {
        edge_props.push(read_edge(&mut cur)?);
    }
    let room_map_len = cur.read_u32::<LittleEndian>()? as usize;
    let mut room_map = Vec::with_capacity(room_map_len);
    for _ in 0..room_map_len {
        room_map.push(cur.read_u32::<LittleEndian>()?);
    }
    let graph = Graph::from_csr(num_nodes, row_ptr, col_idx, node_props, edge_props, room_map)
        .map_err(|e| LevelFileError::StructureMismatch(format!("{e}")))?;

    let num_rooms = cur.read_u32::<LittleEndian>()? as usize;
    let mut rooms_vec = Vec::with_capacity(num_rooms);
    for _ in 0..num_rooms {
        let id = cur.read_u32::<LittleEndian>()?;
        let node_start = cur.read_u32::<LittleEndian>()? as usize;
        let node_count = cur.read_u32::<LittleEndian>()? as usize;
        rooms_vec.push(crate::rooms::Room { id, node_start, node_count });
    }
    let max_room_id = rooms_vec.iter().map(|r| r.id).max().unwrap_or(0) as usize + 1;
    let mut room_index = vec![usize::MAX; max_room_id];
    for (idx, room) in rooms_vec.iter().enumerate() {
        room_index[room.id as usize] = idx;
    }
    let rooms = RoomCollection { rooms: rooms_vec, room_index };

    let n_rel = cur.read_u32::<LittleEndian>()? as usize;
    let mut names = Vec::with_capacity(n_rel);
    for _ in 0..n_rel {
        let len = cur.read_u32::<LittleEndian>()? as usize;
        let mut buf = vec![0u8; len];
        cur.read_exact(&mut buf)?;
        names.push(String::from_utf8_lossy(&buf).into_owned());
    }
    let mut sim = Vec::with_capacity(n_rel * n_rel);
    for _ in 0..(n_rel * n_rel) {
        sim.push(cur.read_f32::<LittleEndian>()?);
    }
    let vocab = RelationVocab::new(names, sim)
        .map_err(|e| LevelFileError::StructureMismatch(format!("{e}")))?;

    Ok((graph, rooms, vocab))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EdgeProps, Graph, NodeProps};
    use crate::rooms::RoomCollection;
    use crate::relation::RelationVocab;
    use std::fs;

    #[test]
    fn roundtrip_v2() {
        let e = (1u32, EdgeProps { attenuation: 0.25, relation: 1, is_portal: false });
        let adj = vec![vec![e], vec![]];
        let mut graph = Graph::from_adjacency(2, adj, NodeProps::default()).unwrap();
        graph.set_room(0, 0).unwrap();
        graph.set_room(1, 1).unwrap();
        graph.set_node_props(0, NodeProps { reflection: 0.5, refraction_index: 2.0 }).unwrap();

        let rooms = RoomCollection::from_room_map(graph.room_map());
        let vocab = RelationVocab::new(
            vec!["isa".into(), "causes".into()],
            vec![1.0, 0.3, 0.3, 1.0],
        ).unwrap();

        let path = "test_level_v2.rgdb";
        write_level_file(&graph, &rooms, &vocab, path).unwrap();
        let (g2, r2, v2) = read_level_file_mmap(path).unwrap();

        assert_eq!(g2.num_nodes(), 2);
        assert_eq!(g2.num_edges(), 1);
        assert!((g2.node_props()[0].reflection - 0.5).abs() < 1e-6);
        assert_eq!(g2.edge_props()[0].relation, 1);
        assert_eq!(r2.num_rooms(), rooms.num_rooms());
        assert_eq!(v2.id_of("causes"), Some(1));
        assert!((v2.similarity(0, 1) - 0.3).abs() < 1e-6);

        let _ = fs::remove_file(path);
    }
}
